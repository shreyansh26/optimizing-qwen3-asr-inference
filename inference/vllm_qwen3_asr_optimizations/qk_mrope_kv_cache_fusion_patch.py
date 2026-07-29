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

# Qwen3-ASR-1.7B attention geometry.  Keep the derived widths next to their
# sources so a kernel launch does not require consulting the model config:
#
#   num_q_heads  = 16
#   num_kv_heads = 8
#   head_dim     = 128 values/head
#
#   q_width         = num_q_heads  * head_dim = 16 * 128 = 2048
#   k_width/v_width = num_kv_heads * head_dim =  8 * 128 = 1024
#   qkv_width       = q_width + k_width + v_width          = 4096
#
# q/k/v are split views of the packed [num_tokens, qkv_width] projection.
# Their logical widths differ, but their usual token stride remains qkv_width.
_NUM_Q_HEADS = 16
_NUM_KV_HEADS = 8
_HEAD_DIM = 128
_HALF_ROTARY_DIM = _HEAD_DIM // 2
_Q_WIDTH = _NUM_Q_HEADS * _HEAD_DIM
_KV_WIDTH = _NUM_KV_HEADS * _HEAD_DIM
_PACKED_QKV_WIDTH = _Q_WIDTH + 2 * _KV_WIDTH
_TOTAL_QK_HEADS = _NUM_Q_HEADS + _NUM_KV_HEADS

# [24 temporal, 20 height, 20 width] counts rotary pairs, not channels:
# 24 + 20 + 20 = 64 pairs and 64 * 2 = the full 128-channel head.
_MROPE_SECTION = (24, 20, 20)
_MROPE_H_END = 3 * _MROPE_SECTION[1]  # 60: interleaved H predicate limit
_MROPE_W_END = 3 * _MROPE_SECTION[2]  # 60: interleaved W predicate limit

# This is a benchmark-selected launch policy, not model geometry.  Below the
# threshold, two larger programs reduce launch count; above it, three smaller
# programs remove padded K-head work and reduce register pressure.
_LARGE_TOKEN_THRESHOLD = 512
_SMALL_HEADS_PER_PROGRAM = 16
_LARGE_HEADS_PER_PROGRAM = 8
_SMALL_NUM_WARPS = 4
_LARGE_NUM_WARPS = 2


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
    programs_per_token: tl.constexpr,
    block_heads: tl.constexpr,
    write_kv_cache: tl.constexpr,
):
    """Process one token/head-group tile.

    Logical tensor contracts (all strides are measured in elements):

      q_ptr:          [num_tokens, num_q_heads * head_dim]
      k_ptr, v_ptr:   [num_tokens, num_kv_heads * head_dim]
      q_out_ptr:      contiguous [num_tokens, num_q_heads * head_dim]
      k_out_ptr:      contiguous [num_tokens, num_kv_heads * head_dim]
      positions_ptr:  [3 axes, num_tokens] in temporal/height/width order
      cache_ptr:      [max_positions, head_dim]
                      = [half_rotary_dim cosine | half_rotary_dim sine]
      norm weights:   [head_dim]
      K/V cache view: [num_blocks, cache_block_size, num_kv_heads, head_dim]

    For this specialization, num_q_heads=16, num_kv_heads=8, head_dim=128,
    and half_rotary_dim=64. The symbolic names above match the kernel arguments.

    Grid = tokens * programs_per_token.  A program owns one token and
    block_heads logical rows from the combined [16 Q heads | 8 K heads] space.
    Triton distributes the resulting [block_heads, 128] tile over num_warps;
    block tensor elements are not CUDA threads.
    """
    program = tl.program_id(0)
    token = program // programs_per_token
    program_in_token = program % programs_per_token
    heads = (
        program_in_token * heads_per_program
        + tl.arange(0, block_heads)[:, None]
    )
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
    heads_per_program: int = _SMALL_HEADS_PER_PROGRAM,
    num_warps: int = _SMALL_NUM_WARPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Q/K head-wise RMSNorm and interleaved-section MRoPE."""
    if positions.ndim != 2 or positions.shape[0] != 3:
        raise ValueError("The fused kernel requires three-axis MRoPE positions")
    if q.shape[-1] != _Q_WIDTH or k.shape[-1] != _KV_WIDTH:
        raise ValueError("The fused kernel is specialized for Qwen3-ASR-1.7B")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise ValueError("The fused kernel currently supports BF16 Q/K inputs")
    if tuple(mrope_section) != _MROPE_SECTION:
        raise ValueError("Unexpected Qwen3-ASR MRoPE section layout")

    q_out = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=q.dtype)
    k_out = torch.empty((k.shape[0], k.shape[1]), device=k.device, dtype=k.dtype)
    programs_per_token = triton.cdiv(_TOTAL_QK_HEADS, heads_per_program)
    block_heads = triton.next_power_of_2(heads_per_program)
    mrope_h_end = 3 * mrope_section[1]
    mrope_w_end = 3 * mrope_section[2]

    # write_kv_cache=False makes every V/cache expression dead code at Triton
    # compile time.  Use named dummy values to make the positional launch
    # contract explicit instead of making the repeated q/1/0 values look real.
    dummy_v_ptr = q
    dummy_cache_ptr = q
    dummy_slot_mapping_ptr = q
    dummy_stride = 1
    num_cache_tokens = 0
    grid = (q.shape[0] * programs_per_token,)
    _qk_norm_mrope_kernel[grid](
        q,
        k,
        dummy_v_ptr,
        q_out,
        k_out,
        dummy_cache_ptr,
        dummy_cache_ptr,
        dummy_slot_mapping_ptr,
        positions,
        cos_sin_cache,
        q_weight,
        k_weight,
        # q/k/v token strides. q is also the compile-time-dead dummy V.
        q.stride(0),
        k.stride(0),
        q.stride(0),
        # Compile-time-dead paged-cache layout.
        dummy_stride,
        dummy_stride,
        dummy_stride,
        dummy_stride,
        num_cache_tokens,
        # positions[axis, token] and cos_sin_cache[position, frequency].
        positions.stride(0),
        positions.stride(1),
        cos_sin_cache.stride(0),
        eps,
        _NUM_Q_HEADS,
        _NUM_KV_HEADS,
        _HEAD_DIM,
        _HALF_ROTARY_DIM,
        mrope_h_end,
        mrope_w_end,
        heads_per_program,
        programs_per_token,
        block_heads,
        False,
        num_warps=num_warps,
    )
    return q_out, k_out


def _select_fused_cache_launch(
    num_tokens: int,
) -> tuple[int, int, int, int]:
    """Return heads/program, programs/token, block heads, and CUDA warps."""
    if num_tokens >= _LARGE_TOKEN_THRESHOLD:
        heads_per_program = _LARGE_HEADS_PER_PROGRAM
        num_warps = _LARGE_NUM_WARPS
    else:
        heads_per_program = _SMALL_HEADS_PER_PROGRAM
        num_warps = _SMALL_NUM_WARPS

    programs_per_token = triton.cdiv(_TOTAL_QK_HEADS, heads_per_program)
    block_heads = triton.next_power_of_2(heads_per_program)
    return heads_per_program, programs_per_token, block_heads, num_warps


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
            and key_cache.shape[-2:] == (_NUM_KV_HEADS, _HEAD_DIM)
            and value_cache.shape[-2:] == (_NUM_KV_HEADS, _HEAD_DIM)
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
            list(_MROPE_SECTION),
        )
        if slot_mapping is not None:
            attn_layer.impl.do_kv_cache_update(
                attn_layer,
                k_out.view(-1, _NUM_KV_HEADS, _HEAD_DIM),
                value.view(-1, _NUM_KV_HEADS, _HEAD_DIM),
                kv_cache,
                slot_mapping,
            )
    else:
        (
            heads_per_program,
            programs_per_token,
            block_heads,
            num_warps,
        ) = _select_fused_cache_launch(q.shape[0])

        # The full cache dimensions are:
        #
        #   [num_cache_blocks, 2 K/V planes, cache_block_size,
        #    num_kv_heads, head_dim]
        #
        # unbind(1) selects the K or V plane. Each resulting view is:
        #
        #   [num_cache_blocks, cache_block_size, num_kv_heads, head_dim]
        #
        # Both views have different base pointers but the same strides. For
        # cache_block_size=16, num_kv_heads=_NUM_KV_HEADS=8, and
        # head_dim=_HEAD_DIM=128, those strides are:
        #
        #   cache_block_stride
        #       = 2 * cache_block_size * num_kv_heads * head_dim
        #       = 2 * 16 * 8 * 128 = 32768
        #       (the factor 2 is the original K/V-plane dimension)
        #   cache_page_stride
        #       = num_kv_heads * head_dim = 8 * 128 = 1024
        #   cache_head_stride
        #       = head_dim = 128
        #
        # Read the live tensor metadata rather than assuming those usual
        # values; value_cache legitimately reuses the key-cache strides.
        cache_block_stride = key_cache.stride(0)
        cache_page_stride = key_cache.stride(1)
        cache_head_stride = key_cache.stride(2)
        cache_block_size = key_cache.shape[1]
        num_cache_tokens = slot_mapping.shape[0]
        grid = (q.shape[0] * programs_per_token,)
        _qk_norm_mrope_kernel[grid](
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
            # q/k/v are packed-QKV split views. At TP=1:
            # q_width=_NUM_Q_HEADS*_HEAD_DIM=2048,
            # kv_width=_NUM_KV_HEADS*_HEAD_DIM=1024, and
            # qkv_width=q_width+2*kv_width=4096. All three views therefore
            # normally have token stride qkv_width=4096.
            q.stride(0),
            k.stride(0),
            value.stride(0),
            cache_block_stride,
            cache_page_stride,
            cache_head_stride,
            cache_block_size,
            num_cache_tokens,
            # positions is [3, tokens]; a contiguous tensor has strides
            # [tokens, 1], but sliced inputs retain their real backing strides.
            positions.stride(0),
            positions.stride(1),
            # cos_sin_cache is [max_positions, 128], normally stride(0)=128.
            cos_sin_cache.stride(0),
            eps,
            _NUM_Q_HEADS,
            _NUM_KV_HEADS,
            _HEAD_DIM,
            _HALF_ROTARY_DIM,
            _MROPE_H_END,
            _MROPE_W_END,
            heads_per_program,
            programs_per_token,
            block_heads,
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
            and getattr(rotary, "mrope_section", None) == list(_MROPE_SECTION)
            and self.num_heads == _NUM_Q_HEADS
            and self.num_kv_heads == _NUM_KV_HEADS
            and self.head_dim == _HEAD_DIM
            and self.q_size + 2 * self.kv_size == _PACKED_QKV_WIDTH
        )
        if not compatible:
            return original_forward(self, positions, hidden_states)

        qkv, _ = self.qkv_proj(hidden_states)
        # qkv is [num_tokens, qkv_width], where:
        #   qkv_width = (_NUM_Q_HEADS + 2 * _NUM_KV_HEADS) * _HEAD_DIM
        #             = (16 + 2 * 8) * 128 = 4096.
        # split() returns views, so q/k/v keep qkv.stride(0)=qkv_width.
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
            q.view(-1, _NUM_Q_HEADS, _HEAD_DIM),
            k.view(-1, _NUM_KV_HEADS, _HEAD_DIM),
            v.view(-1, _NUM_KV_HEADS, _HEAD_DIM),
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
