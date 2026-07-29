"""Fuse Q/K RMSNorm, multi-axis RoPE, and KV-cache writes."""

from __future__ import annotations

from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)


logger = init_logger("vllm.qwen3_asr_qk_mrope")


@triton.jit
def _qk_norm_mrope_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    q_out_ptr,
    k_out_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    positions_ptr,
    cache_ptr,
    q_weight_ptr,
    k_weight_ptr,
    q_token_stride: tl.constexpr,
    k_token_stride: tl.constexpr,
    v_token_stride: tl.constexpr,
    cache_block_stride: tl.constexpr,
    cache_page_stride: tl.constexpr,
    cache_head_stride: tl.constexpr,
    cache_block_size: tl.constexpr,
    num_cache_tokens,
    position_axis_stride,
    position_token_stride,
    cache_position_stride: tl.constexpr,
    eps: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    half_rotary_dim: tl.constexpr,
    mrope_h_end: tl.constexpr,
    mrope_w_end: tl.constexpr,
    heads_per_program: tl.constexpr,
    groups_per_token: tl.constexpr,
    block_heads: tl.constexpr,
    write_kv_cache: tl.constexpr,
):
    program = tl.program_id(0)
    token = program // groups_per_token
    head_group = program % groups_per_token
    heads = head_group * heads_per_program + tl.arange(0, block_heads)[:, None]
    dims = tl.arange(0, head_dim)[None, :]
    valid_head = heads < num_q_heads + num_kv_heads
    is_q = heads < num_q_heads
    local_head = tl.where(is_q, heads, heads - num_q_heads)

    q_input = q_ptr + token * q_token_stride + local_head * head_dim + dims
    k_input = k_ptr + token * k_token_stride + local_head * head_dim + dims
    input_ptrs = tl.where(is_q, q_input, k_input)
    values = tl.load(input_ptrs, mask=valid_head, other=0.0).to(tl.float32)

    # Match the decode graph's Inductor reduction (R0_BLOCK=64): accumulate
    # dimensions d and d + 64 first, then reduce the resulting 64 lanes.
    squared_values = tl.reshape(
        values * values,
        (block_heads, 2, half_rotary_dim),
    )
    squared_mean = tl.sum(tl.sum(squared_values, axis=1), axis=1) / head_dim
    inv_rms = tl.rsqrt(squared_mean + eps)
    q_weights = q_weight_ptr + dims
    k_weights = k_weight_ptr + dims
    weights = tl.load(tl.where(is_q, q_weights, k_weights), mask=valid_head)
    # Inductor fuses the RMSNorm epilogue into its MRoPE kernel, so this
    # intermediate remains FP32 in the control graph.
    normalized = values * inv_rms[:, None] * weights.to(tl.float32)

    frequency = dims % half_rotary_dim
    h_axis = (frequency % 3 == 1) & (frequency <= mrope_h_end)
    w_axis = (frequency % 3 == 2) & (frequency <= mrope_w_end)
    axis = tl.where(h_axis, 1, tl.where(w_axis, 2, 0))
    position = tl.load(
        positions_ptr
        + axis * position_axis_stride
        + token * position_token_stride
    )
    cache_base = position * cache_position_stride + frequency
    cosine = tl.load(cache_ptr + cache_base).to(tl.float32)
    sine = tl.load(cache_ptr + cache_base + half_rotary_dim).to(tl.float32)

    partner_dims = (dims + half_rotary_dim) % head_dim
    partner_q = q_ptr + token * q_token_stride + local_head * head_dim + partner_dims
    partner_k = k_ptr + token * k_token_stride + local_head * head_dim + partner_dims
    partner_values = tl.load(
        tl.where(is_q, partner_q, partner_k), mask=valid_head, other=0.0
    ).to(tl.float32)
    partner_q_weight = q_weight_ptr + partner_dims
    partner_k_weight = k_weight_ptr + partner_dims
    partner_weights = tl.load(
        tl.where(is_q, partner_q_weight, partner_k_weight), mask=valid_head
    ).to(tl.float32)
    partner_normalized = partner_values * inv_rms[:, None] * partner_weights

    sign = tl.where(dims < half_rotary_dim, -1.0, 1.0)
    rotated = normalized * cosine + sign * partner_normalized * sine

    q_output = (
        q_out_ptr + token * (num_q_heads * head_dim) + local_head * head_dim + dims
    )
    k_output = (
        k_out_ptr + token * (num_kv_heads * head_dim) + local_head * head_dim + dims
    )
    tl.store(tl.where(is_q, q_output, k_output), rotated, mask=valid_head)

    if write_kv_cache:
        slot = tl.load(
            slot_mapping_ptr + token,
            mask=token < num_cache_tokens,
            other=-1,
        ).to(tl.int64)
        valid_cache = valid_head & ~is_q & (slot >= 0)
        block = slot // cache_block_size
        block_offset = slot % cache_block_size
        cache_offset = (
            block * cache_block_stride
            + block_offset * cache_page_stride
            + local_head * cache_head_stride
            + dims
        )
        tl.store(key_cache_ptr + cache_offset, rotated, mask=valid_cache)
        value = tl.load(
            v_ptr + token * v_token_stride + local_head * head_dim + dims,
            mask=valid_cache,
            other=0.0,
        )
        tl.store(value_cache_ptr + cache_offset, value, mask=valid_cache)


def fused_qk_norm_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    mrope_section: list[int],
    heads_per_program: int = 16,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Q/K head-wise RMSNorm and interleaved-section MRoPE."""
    if positions.ndim != 2 or positions.shape[0] != 3:
        raise ValueError("The fused kernel requires three-axis MRoPE positions")
    if q.shape[-1] != 2048 or k.shape[-1] != 1024:
        raise ValueError("The fused kernel is specialized for Qwen3-ASR-1.7B")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise ValueError("The fused kernel currently supports BF16 Q/K inputs")
    if mrope_section != [24, 20, 20]:
        raise ValueError("Unexpected Qwen3-ASR MRoPE section layout")

    q_out = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=q.dtype)
    k_out = torch.empty((k.shape[0], k.shape[1]), device=k.device, dtype=k.dtype)
    total_heads = 24
    groups_per_token = triton.cdiv(total_heads, heads_per_program)
    block_heads = triton.next_power_of_2(heads_per_program)
    _qk_norm_mrope_kernel[(q.shape[0] * groups_per_token,)](
        q,
        k,
        q,
        q_out,
        k_out,
        q,
        q,
        q,
        positions,
        cos_sin_cache,
        q_weight,
        k_weight,
        q.stride(0),
        k.stride(0),
        q.stride(0),
        1,
        1,
        1,
        1,
        0,
        positions.stride(0),
        positions.stride(1),
        cos_sin_cache.stride(0),
        eps,
        16,
        8,
        128,
        64,
        3 * mrope_section[1],
        3 * mrope_section[2],
        heads_per_program,
        groups_per_token,
        block_heads,
        False,
        num_warps=num_warps,
    )
    return q_out, k_out


def fused_qk_norm_mrope_kv_update_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    layer_name: LayerNameType,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse Q/K preprocessing with the native-BF16 paged-cache scatter."""
    layer_name = _resolve_layer_name(layer_name)
    _, attn_layer, kv_cache, slot_mapping = get_attention_context(layer_name)

    q_out = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=q.dtype)
    k_out = torch.empty((k.shape[0], k.shape[1]), device=k.device, dtype=k.dtype)
    compatible_cache = False
    if slot_mapping is not None and kv_cache.ndim >= 2:
        key_cache, value_cache = kv_cache.unbind(1)
        compatible_cache = (
            key_cache.ndim == 4
            and value_cache.ndim == 4
            and key_cache.dtype == q.dtype
            and value_cache.dtype == value.dtype
            and key_cache.shape[-2:] == (8, 128)
            and value_cache.shape[-2:] == (8, 128)
        )

    if not compatible_cache:
        q_out, k_out = fused_qk_norm_mrope(
            q,
            k,
            positions,
            cos_sin_cache,
            q_weight,
            k_weight,
            eps,
            [24, 20, 20],
        )
        if slot_mapping is not None:
            attn_layer.impl.do_kv_cache_update(
                attn_layer,
                k_out.view(-1, 8, 128),
                value.view(-1, 8, 128),
                kv_cache,
                slot_mapping,
            )
    else:
        if q.shape[0] >= 512:
            heads_per_program = 8
            groups_per_token = 3
            num_warps = 2
        else:
            heads_per_program = 16
            groups_per_token = 2
            num_warps = 4
        _qk_norm_mrope_kernel[(q.shape[0] * groups_per_token,)](
            q,
            k,
            value,
            q_out,
            k_out,
            key_cache,
            value_cache,
            slot_mapping,
            positions,
            cos_sin_cache,
            q_weight,
            k_weight,
            q.stride(0),
            k.stride(0),
            value.stride(0),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.shape[1],
            slot_mapping.shape[0],
            positions.stride(0),
            positions.stride(1),
            cos_sin_cache.stride(0),
            eps,
            16,
            8,
            128,
            64,
            60,
            60,
            heads_per_program,
            groups_per_token,
            heads_per_program,
            True,
            num_warps=num_warps,
        )

    dummy = torch.empty(0, device=kv_cache.device, dtype=kv_cache.dtype)
    return q_out, k_out, dummy


def fused_qk_norm_mrope_kv_update_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    layer_name: LayerNameType,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del value, positions, cos_sin_cache, q_weight, k_weight, eps, layer_name
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty(0, device=q.device, dtype=q.dtype),
    )


direct_register_custom_op(
    op_name="asr_qk_norm_mrope_kv_update",
    op_func=fused_qk_norm_mrope_kv_update_impl,
    fake_impl=fused_qk_norm_mrope_kv_update_fake,
)


def install_qk_mrope_kv_cache_fusion_patch() -> None:
    """Patch the Qwen3 attention forward only for compatible ASR layers."""
    from vllm.model_executor.models import qwen3

    original_forward = qwen3.Qwen3Attention.forward
    if getattr(original_forward, "_asr_qk_mrope_kv_cache_fusion", False):
        return

    def patched_forward(
        self: Any,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        rotary = self.rotary_emb
        compatible = (
            positions.ndim == 2
            and getattr(rotary, "mrope_interleaved", False)
            and getattr(rotary, "mrope_section", None) == [24, 20, 20]
            and self.num_heads == 16
            and self.num_kv_heads == 8
            and self.head_dim == 128
        )
        if not compatible:
            return original_forward(self, positions, hidden_states)

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        cache = rotary._match_cos_sin_cache_dtype(q)
        encoded_layer_name = _encode_layer_name(self.attn.layer_name)
        q, k, kv_cache_dummy = torch.ops.vllm.asr_qk_norm_mrope_kv_update(
            q,
            k,
            v,
            positions,
            cache,
            self.q_norm.weight,
            self.k_norm.weight,
            self.q_norm.variance_epsilon,
            encoded_layer_name,
        )
        output = torch.empty(
            (q.shape[0], self.num_heads, self.head_dim),
            dtype=q.dtype,
            device=q.device,
        )
        torch.ops.vllm.unified_attention_with_output(
            q.view(-1, self.num_heads, self.head_dim),
            k.view(-1, self.num_kv_heads, self.head_dim),
            v.view(-1, self.num_kv_heads, self.head_dim),
            output,
            encoded_layer_name,
            kv_cache_dummy_dep=kv_cache_dummy,
        )
        attn_output = output.view(q.shape[0], -1)
        output, _ = self.o_proj(attn_output)
        return output

    patched_forward._asr_qk_mrope_kv_cache_fusion = True
    qwen3.Qwen3Attention.forward = patched_forward
    logger.info("Installed Qwen3-ASR Q/K RMSNorm + MRoPE fusion patch")
