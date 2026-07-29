"""Avoid GPU scalar readbacks in the supported Qwen3-ASR audio encoder."""

from __future__ import annotations

from functools import wraps
import os
from typing import Any

import torch


ENV_NAME = "ASR_AUDIO_CPU_MAXSEQLEN"
STATIC_MAX_SEQLEN = 104
_STATIC_MAX_SEQLEN_CPU = torch.tensor(
    STATIC_MAX_SEQLEN,
    dtype=torch.int32,
    device="cpu",
)
_PATCH_MARKER = "_asr_audio_cpu_maxseqlen_patch"


def audio_cpu_maxseqlen_enabled() -> bool:
    """Return the strict boolean environment gate for this experiment."""
    raw_value = os.environ.get(ENV_NAME, "0")
    if raw_value not in {"0", "1"}:
        raise ValueError(f"{ENV_NAME} must be 0 or 1, got {raw_value!r}")
    return raw_value == "1"


def _has_expected_conv_geometry(module: Any, in_channels: int) -> bool:
    return (
        getattr(module, "in_channels", None) == in_channels
        and getattr(module, "out_channels", None) == 480
        and getattr(module, "kernel_size", None) == (3, 3)
        and getattr(module, "stride", None) == (2, 2)
        and getattr(module, "padding", None) == (1, 1)
        and getattr(module, "dilation", None) == (1, 1)
        and getattr(module, "groups", None) == 1
    )


def _is_supported_encoder(
    encoder: Any,
    *,
    model_cls: type[Any],
    flash_backend: Any,
) -> bool:
    if type(encoder) is not model_cls or encoder.attn_backend != flash_backend:
        return False
    if (
        getattr(encoder, "n_window", None) != 50
        or getattr(encoder, "n_window_infer", None) != 800
        or getattr(encoder, "num_mel_bins", None) != 128
        or getattr(encoder, "max_source_positions", None) != 1500
    ):
        return False

    layers = getattr(encoder, "layers", ())
    if len(layers) != 24:
        return False
    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        if (
            getattr(layer, "embed_dim", None) != 1024
            or getattr(attention, "embed_dim", None) != 1024
            or getattr(attention, "num_heads", None) != 16
            or getattr(attention, "head_dim", None) != 64
        ):
            return False

    return (
        _has_expected_conv_geometry(encoder.conv2d1, 1)
        and _has_expected_conv_geometry(encoder.conv2d2, 480)
        and _has_expected_conv_geometry(encoder.conv2d3, 480)
    )


def install_audio_cpu_maxseqlen_patch(
    *,
    model_cls: type[Any] | None = None,
    flash_backend: Any | None = None,
) -> bool:
    """Return a cached CPU upper bound only for the exact supported encoder."""
    if not audio_cpu_maxseqlen_enabled():
        return False

    if model_cls is None:
        from vllm.model_executor.models.qwen3_asr import (
            Qwen3OmniMoeAudioEncoder,
        )

        model_cls = Qwen3OmniMoeAudioEncoder
    if flash_backend is None:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        flash_backend = AttentionBackendEnum.FLASH_ATTN

    original_method = model_cls.compute_attn_mask_seqlen
    if getattr(original_method, _PATCH_MARKER, False):
        return True

    @wraps(original_method)
    def patched_compute_attn_mask_seqlen(
        self: Any,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor | None:
        compatible_input = (
            isinstance(cu_seqlens, torch.Tensor)
            and cu_seqlens.ndim == 1
            and cu_seqlens.dtype == torch.int32
            and cu_seqlens.is_cuda
            and cu_seqlens.numel() >= 2
        )
        if compatible_input and _is_supported_encoder(
            self,
            model_cls=model_cls,
            flash_backend=flash_backend,
        ):
            return _STATIC_MAX_SEQLEN_CPU
        return original_method(self, cu_seqlens)

    setattr(patched_compute_attn_mask_seqlen, _PATCH_MARKER, True)
    model_cls.compute_attn_mask_seqlen = patched_compute_attn_mask_seqlen
    return True
