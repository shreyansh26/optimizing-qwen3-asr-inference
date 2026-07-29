from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import torch


PATCH_DIR = (
    Path(__file__).resolve().parents[1]
    / "inference"
    / "vllm_qwen3_asr_optimizations"
)
sys.path.insert(0, str(PATCH_DIR))

from audio_cpu_metadata_pack_patch import (  # noqa: E402
    ENV_NAME,
    MAX_SEQLEN_ENV_NAME,
    _EXPECTED_VLLM_VERSION,
    _EXPECTED_VLLM_WHEEL_HASH,
    _build_cpu_metadata,
    _expected_audio_output_lengths,
    _installed_vllm_wheel_is_supported,
    _make_cpu_field_config,
    _make_patched_forward,
    audio_cpu_metadata_pack_enabled,
    install_audio_cpu_metadata_pack_patch,
    pack_valid_rows,
)


class AudioCpuMetadataPackPatchTest(unittest.TestCase):
    def test_environment_gate_is_strict(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(audio_cpu_metadata_pack_enabled())
        with patch.dict(os.environ, {ENV_NAME: "1"}, clear=True):
            self.assertTrue(audio_cpu_metadata_pack_enabled())
        for value in ("", "true", "2", "-1"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {ENV_NAME: value}, clear=True):
                    with self.assertRaises(ValueError):
                        audio_cpu_metadata_pack_enabled()

    def test_cpu_metadata_matches_chunk_and_whole_audio_lengths(self) -> None:
        feature_lens = torch.tensor([100, 101, 250], dtype=torch.int64)
        aftercnn_lens = _expected_audio_output_lengths(feature_lens)
        self.assertEqual(aftercnn_lens.tolist(), [13, 14, 33])

        chunk_lengths, pack_metadata, cu_seqlens = _build_cpu_metadata(
            feature_lens,
            aftercnn_lens,
            n_window=50,
            n_window_infer=800,
        )

        self.assertEqual(chunk_lengths.tolist(), [100, 100, 1, 100, 100, 50])
        self.assertEqual(pack_metadata.dtype, torch.int32)
        self.assertEqual(pack_metadata.device.type, "cpu")
        self.assertEqual(
            pack_metadata.tolist(),
            [13, 13, 1, 13, 13, 7, 0, 13, 26, 27, 40, 53],
        )
        self.assertEqual(cu_seqlens, [0, 13, 27, 60])

    def test_cpu_metadata_accepts_int32_lengths(self) -> None:
        feature_lens = torch.tensor([100, 101, 250], dtype=torch.int32)
        aftercnn_lens = _expected_audio_output_lengths(feature_lens)

        chunk_lengths, pack_metadata, cu_seqlens = _build_cpu_metadata(
            feature_lens,
            aftercnn_lens,
            n_window=50,
            n_window_infer=800,
        )

        self.assertEqual(chunk_lengths.dtype, torch.int64)
        self.assertEqual(chunk_lengths.tolist(), [100, 100, 1, 100, 100, 50])
        self.assertEqual(
            pack_metadata.tolist(),
            [13, 13, 1, 13, 13, 7, 0, 13, 26, 27, 40, 53],
        )
        self.assertEqual(cu_seqlens, [0, 13, 27, 60])

    def test_cpu_metadata_handles_short_window_geometry(self) -> None:
        feature_lens = torch.tensor([1, 50, 99], dtype=torch.int64)
        aftercnn_lens = _expected_audio_output_lengths(feature_lens)
        self.assertEqual(aftercnn_lens.tolist(), [1, 7, 13])

        chunk_lengths, pack_metadata, cu_seqlens = _build_cpu_metadata(
            feature_lens,
            aftercnn_lens,
            n_window=50,
            n_window_infer=800,
        )
        self.assertEqual(chunk_lengths.tolist(), [1, 50, 99])
        self.assertEqual(pack_metadata.tolist(), [1, 7, 13, 0, 1, 8])
        self.assertEqual(cu_seqlens, [0, 1, 8, 21])

    def test_mismatched_whole_audio_lengths_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "whole-audio"):
            _build_cpu_metadata(
                torch.tensor([100], dtype=torch.int64),
                torch.tensor([12], dtype=torch.int64),
                n_window=50,
                n_window_infer=800,
            )

    def test_field_config_keeps_only_audio_lengths_on_cpu(self) -> None:
        from vllm.multimodal.inputs import MultiModalFieldConfig

        feature_config = MultiModalFieldConfig.batched("audio")

        def original(_inputs):
            return {
                "input_audio_features": feature_config,
                "audio_feature_lengths": MultiModalFieldConfig.batched("audio"),
            }

        patched = _make_cpu_field_config(original, MultiModalFieldConfig)
        configs = patched({"audio_feature_lengths": torch.tensor([100])})

        self.assertIs(configs["input_audio_features"], feature_config)
        self.assertTrue(configs["audio_feature_lengths"].field.keep_on_cpu)
        self.assertEqual(configs["audio_feature_lengths"].modality, "audio")

    def test_cpu_pack_rejects_non_cuda_input(self) -> None:
        padded = torch.empty((3, 13, 1024), dtype=torch.bfloat16)
        metadata = torch.tensor([13, 7, 1, 0, 13, 20], dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "Unsupported tensor layout"):
            pack_valid_rows(
                padded,
                metadata,
                async_tensor_h2d=Mock(),
            )

    def test_forward_falls_back_without_supported_cuda_input(self) -> None:
        seen = {}

        def original(_encoder, inputs, feature_lens, aftercnn_lens):
            seen["args"] = (inputs, feature_lens, aftercnn_lens)
            return torch.tensor([7])

        class FakeEncoder:
            pass

        patched = _make_patched_forward(
            original,
            model_cls=FakeEncoder,
            flash_backend=object(),
            async_tensor_h2d=Mock(),
        )
        encoder = FakeEncoder()
        inputs = torch.empty((128, 4), dtype=torch.bfloat16)
        feature_lens = torch.tensor([4])
        aftercnn_lens = torch.tensor([1])

        result = patched(encoder, inputs, feature_lens, aftercnn_lens)

        self.assertEqual(result.item(), 7)
        self.assertIs(seen["args"][0], inputs)
        self.assertIs(seen["args"][1], feature_lens)
        self.assertIs(seen["args"][2], aftercnn_lens)

    def test_current_installed_vllm_wheel_matches_guard(self) -> None:
        self.assertTrue(_installed_vllm_wheel_is_supported())

    def test_vllm_version_drift_is_rejected(self) -> None:
        distribution = SimpleNamespace(version="0.0.0")
        with patch(
            "audio_cpu_metadata_pack_patch.importlib_metadata.distribution",
            return_value=distribution,
        ):
            self.assertFalse(_installed_vllm_wheel_is_supported())

    def test_missing_max_seqlen_prerequisite_does_not_patch(self) -> None:
        from vllm.model_executor.models import qwen3_asr
        from vllm.model_executor.models.qwen3_asr import (
            Qwen3OmniMoeAudioEncoder,
        )

        original_forward = Qwen3OmniMoeAudioEncoder.forward
        original_field_config = qwen3_asr._qwen3asr_field_config
        with patch.dict(os.environ, {ENV_NAME: "1"}, clear=True):
            self.assertFalse(install_audio_cpu_metadata_pack_patch())

        self.assertIs(Qwen3OmniMoeAudioEncoder.forward, original_forward)
        self.assertIs(qwen3_asr._qwen3asr_field_config, original_field_config)
        self.assertNotIn(MAX_SEQLEN_ENV_NAME, os.environ)

    def test_wheel_hash_drift_is_rejected(self) -> None:
        wheel_url = "https://example.invalid/vllm.whl"
        distribution = SimpleNamespace(
            version=_EXPECTED_VLLM_VERSION,
            read_text=lambda name: (
                '{"url": "' + wheel_url + '"}'
                if name == "direct_url.json"
                else None
            ),
        )
        with TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "uv.lock"
            lock_path.write_text(
                "\n".join(
                    (
                        "[[package]]",
                        'name = "vllm"',
                        f'version = "{_EXPECTED_VLLM_VERSION}"',
                        "wheels = [",
                        "  { "
                        f'url = "{wheel_url}", '
                        'hash = "sha256:not-the-expected-wheel" '
                        "},",
                        "]",
                    )
                )
            )
            with patch(
                "audio_cpu_metadata_pack_patch.importlib_metadata.distribution",
                return_value=distribution,
            ):
                self.assertFalse(
                    _installed_vllm_wheel_is_supported(_lock_path=lock_path)
                )

        self.assertTrue(_EXPECTED_VLLM_WHEEL_HASH.startswith("sha256:"))

    def test_wheel_mismatch_does_not_install_max_seqlen_prerequisite(self) -> None:
        with (
            patch.dict(
                os.environ,
                {ENV_NAME: "1", MAX_SEQLEN_ENV_NAME: "1"},
                clear=True,
            ),
            patch(
                "audio_cpu_metadata_pack_patch._installed_vllm_wheel_is_supported",
                return_value=False,
            ),
            patch(
                "audio_cpu_metadata_pack_patch.install_audio_cpu_maxseqlen_patch"
            ) as install_max_seqlen,
        ):
            self.assertFalse(install_audio_cpu_metadata_pack_patch())

        install_max_seqlen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
