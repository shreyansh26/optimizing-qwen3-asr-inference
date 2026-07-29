from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import torch


PATCH_DIR = (
    Path(__file__).resolve().parents[1]
    / "inference"
    / "vllm_qwen3_asr_optimizations"
)
sys.path.insert(0, str(PATCH_DIR))

from audio_cpu_maxseqlen_patch import (  # noqa: E402
    ENV_NAME,
    STATIC_MAX_SEQLEN,
    install_audio_cpu_maxseqlen_patch,
)


def _conv(in_channels: int) -> SimpleNamespace:
    return SimpleNamespace(
        in_channels=in_channels,
        out_channels=480,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
    )


def _layers() -> list[SimpleNamespace]:
    attention = lambda: SimpleNamespace(  # noqa: E731
        embed_dim=1024,
        num_heads=16,
        head_dim=64,
    )
    return [
        SimpleNamespace(embed_dim=1024, self_attn=attention()) for _ in range(24)
    ]


def _populate_supported_encoder(encoder: object, backend: object) -> None:
    encoder.attn_backend = backend
    encoder.n_window = 50
    encoder.n_window_infer = 800
    encoder.num_mel_bins = 128
    encoder.max_source_positions = 1500
    encoder.layers = _layers()
    encoder.conv2d1 = _conv(1)
    encoder.conv2d2 = _conv(480)
    encoder.conv2d3 = _conv(480)


def _fake_cuda_cu_seqlens() -> torch.Tensor:
    tensor = Mock(spec=torch.Tensor)
    tensor.ndim = 1
    tensor.dtype = torch.int32
    tensor.is_cuda = True
    tensor.numel.return_value = 3
    return tensor


def _fake_encoder_class() -> type:
    class FakeEncoder:
        def __init__(self, backend: object) -> None:
            self.original_calls = 0
            _populate_supported_encoder(self, backend)

        def compute_attn_mask_seqlen(
            self, cu_seqlens: torch.Tensor
        ) -> torch.Tensor:
            del cu_seqlens
            self.original_calls += 1
            return torch.tensor(7, dtype=torch.int32)

    return FakeEncoder


class AudioCpuMaxSeqlenPatchTest(unittest.TestCase):
    def test_unset_gate_does_not_patch_class(self) -> None:
        model_cls = _fake_encoder_class()
        original = model_cls.compute_attn_mask_seqlen

        with patch.dict(os.environ, {}, clear=True):
            installed = install_audio_cpu_maxseqlen_patch(
                model_cls=model_cls,
                flash_backend=object(),
            )

        self.assertFalse(installed)
        self.assertIs(model_cls.compute_attn_mask_seqlen, original)

    def test_supported_encoder_returns_one_cached_cpu_scalar(self) -> None:
        model_cls = _fake_encoder_class()
        backend = object()
        with patch.dict(os.environ, {ENV_NAME: "1"}, clear=True):
            self.assertTrue(
                install_audio_cpu_maxseqlen_patch(
                    model_cls=model_cls,
                    flash_backend=backend,
                )
            )

        encoder = model_cls(backend)
        first = encoder.compute_attn_mask_seqlen(_fake_cuda_cu_seqlens())
        second = encoder.compute_attn_mask_seqlen(_fake_cuda_cu_seqlens())

        self.assertIs(first, second)
        self.assertEqual(first.item(), STATIC_MAX_SEQLEN)
        self.assertEqual(first.dtype, torch.int32)
        self.assertEqual(first.device.type, "cpu")
        self.assertEqual(encoder.original_calls, 0)

    def test_any_guard_mismatch_falls_back(self) -> None:
        mutations = (
            lambda encoder: setattr(encoder, "n_window", 51),
            lambda encoder: setattr(encoder, "attn_backend", object()),
            lambda encoder: setattr(encoder.conv2d2, "stride", (1, 1)),
            lambda encoder: encoder.layers.pop(),
            lambda encoder: setattr(encoder.layers[0].self_attn, "num_heads", 8),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                model_cls = _fake_encoder_class()
                backend = object()
                with patch.dict(os.environ, {ENV_NAME: "1"}, clear=True):
                    install_audio_cpu_maxseqlen_patch(
                        model_cls=model_cls,
                        flash_backend=backend,
                    )
                encoder = model_cls(backend)
                mutate(encoder)
                result = encoder.compute_attn_mask_seqlen(
                    _fake_cuda_cu_seqlens()
                )
                self.assertEqual(result.item(), 7)
                self.assertEqual(encoder.original_calls, 1)

    def test_non_cuda_input_falls_back(self) -> None:
        model_cls = _fake_encoder_class()
        backend = object()
        with patch.dict(os.environ, {ENV_NAME: "1"}, clear=True):
            install_audio_cpu_maxseqlen_patch(
                model_cls=model_cls,
                flash_backend=backend,
            )

        encoder = model_cls(backend)
        result = encoder.compute_attn_mask_seqlen(
            torch.tensor([0, 4], dtype=torch.int32)
        )
        self.assertEqual(result.item(), 7)
        self.assertEqual(encoder.original_calls, 1)

    def test_install_is_idempotent(self) -> None:
        model_cls = _fake_encoder_class()
        backend = object()
        with patch.dict(os.environ, {ENV_NAME: "1"}, clear=True):
            self.assertTrue(
                install_audio_cpu_maxseqlen_patch(
                    model_cls=model_cls,
                    flash_backend=backend,
                )
            )
            installed_method = model_cls.compute_attn_mask_seqlen
            self.assertTrue(
                install_audio_cpu_maxseqlen_patch(
                    model_cls=model_cls,
                    flash_backend=backend,
                )
            )
        self.assertIs(model_cls.compute_attn_mask_seqlen, installed_method)

    def test_invalid_gate_is_rejected(self) -> None:
        for value in ("", "true", "2", "-1"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {ENV_NAME: value}, clear=True):
                    with self.assertRaises(ValueError):
                        install_audio_cpu_maxseqlen_patch(
                            model_cls=_fake_encoder_class(),
                            flash_backend=object(),
                        )

    def test_104_bounds_every_short_tail_after_three_convolutions(self) -> None:
        windows_per_attention_group = 800 // (50 * 2)
        self.assertEqual(windows_per_attention_group, 8)
        for raw_frames in range(1, 101):
            after_cnn = raw_frames
            for _ in range(3):
                after_cnn = (after_cnn - 1) // 2 + 1
            self.assertLessEqual(
                after_cnn * windows_per_attention_group,
                STATIC_MAX_SEQLEN,
            )
        self.assertEqual(after_cnn, 13)
        self.assertEqual(after_cnn * windows_per_attention_group, 104)

    def test_installed_class_integration_and_idempotence(self) -> None:
        from vllm.model_executor.models.qwen3_asr import (
            Qwen3OmniMoeAudioEncoder,
        )
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        with patch.dict(os.environ, {ENV_NAME: "1"}, clear=True):
            self.assertTrue(install_audio_cpu_maxseqlen_patch())
            installed_method = Qwen3OmniMoeAudioEncoder.compute_attn_mask_seqlen
            self.assertTrue(install_audio_cpu_maxseqlen_patch())

        self.assertIs(
            Qwen3OmniMoeAudioEncoder.compute_attn_mask_seqlen,
            installed_method,
        )
        encoder = Qwen3OmniMoeAudioEncoder.__new__(Qwen3OmniMoeAudioEncoder)
        torch.nn.Module.__init__(encoder)
        _populate_supported_encoder(encoder, AttentionBackendEnum.FLASH_ATTN)
        result = encoder.compute_attn_mask_seqlen(_fake_cuda_cu_seqlens())
        self.assertEqual(result.item(), STATIC_MAX_SEQLEN)
        self.assertEqual(result.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
