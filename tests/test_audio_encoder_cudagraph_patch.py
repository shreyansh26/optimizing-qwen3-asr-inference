from __future__ import annotations

import os
from pathlib import Path
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

import audio_encoder_cudagraph_patch as combined_patch  # noqa: E402
from audio_cpu_metadata_pack_patch import (  # noqa: E402
    ENV_NAME as METADATA_ENV_NAME,
    MAX_SEQLEN_ENV_NAME,
    _PATCH_MARKER as METADATA_PATCH_MARKER,
    _make_patched_forward,
)
from audio_frontend_cudagraph_patch import (  # noqa: E402
    ENV_NAME as FRONTEND_ENV_NAME,
    _PATCH_MARKER as FRONTEND_PATCH_MARKER,
    install_audio_frontend_cudagraph_patch,
)
from audio_transformer_cudagraph_patch import (  # noqa: E402
    ENV_NAME as TRANSFORMER_ENV_NAME,
    _PATCH_MARKER as TRANSFORMER_PATCH_MARKER,
    install_audio_transformer_cudagraph_patch,
)


def _combined_environment() -> dict[str, str]:
    return {
        MAX_SEQLEN_ENV_NAME: "1",
        METADATA_ENV_NAME: "1",
        FRONTEND_ENV_NAME: "1",
        TRANSFORMER_ENV_NAME: "1",
    }


class AudioEncoderCudagraphPatchTest(unittest.TestCase):
    def test_metadata_forward_calls_frontend_then_transformer_once(self) -> None:
        events = []

        class _FakeEncoder:
            n_window = 50
            n_window_infer = 800

            def compute_attn_mask_seqlen(self, cu_seqlens):
                events.append(("max_seqlen", tuple(cu_seqlens.tolist())))
                return torch.tensor(1, dtype=torch.int32)

        def original_forward(*args):
            del args
            raise AssertionError("supported input must not use original forward")

        def frontend_runner(
            encoder,
            padded_feature,
            chunk_lengths,
            pack_metadata,
            cu_seqlens_values,
            feature_lens_values,
            aftercnn_lens_values,
            *,
            async_tensor_h2d,
        ):
            del encoder, async_tensor_h2d
            events.append(
                (
                    "frontend",
                    tuple(padded_feature.shape),
                    tuple(chunk_lengths.tolist()),
                    tuple(pack_metadata.tolist()),
                    cu_seqlens_values,
                    feature_lens_values,
                    aftercnn_lens_values,
                )
            )
            return torch.full((1, 1024), 2, dtype=torch.bfloat16)

        def async_tensor_h2d(values, *, dtype, device):
            events.append(("metadata_h2d", tuple(values), dtype, device.type))
            return torch.tensor(values, dtype=dtype, device=device)

        def transformer_runner(
            encoder,
            hidden_states,
            cu_seqlens,
            max_seqlen,
            *,
            cu_seqlens_values,
        ):
            del encoder
            events.append(
                (
                    "transformer",
                    tuple(hidden_states.shape),
                    tuple(cu_seqlens.tolist()),
                    int(max_seqlen.item()),
                    cu_seqlens_values,
                )
            )
            return hidden_states + 1

        patched_forward = _make_patched_forward(
            original_forward,
            model_cls=_FakeEncoder,
            flash_backend=object(),
            async_tensor_h2d=async_tensor_h2d,
            frontend_runner=frontend_runner,
            transformer_runner=transformer_runner,
        )
        input_features = torch.arange(
            128 * 2,
            dtype=torch.bfloat16,
        ).reshape(128, 2)
        feature_lens = torch.tensor([2], dtype=torch.int64)
        aftercnn_lens = torch.tensor([1], dtype=torch.int64)

        with (
            patch(
                "audio_cpu_metadata_pack_patch._cpu_input_is_supported",
                return_value=True,
            ),
            patch(
                "audio_cpu_metadata_pack_patch._build_cpu_metadata",
                return_value=(
                    torch.tensor([2], dtype=torch.int64),
                    torch.tensor([1, 0], dtype=torch.int32),
                    [0, 1],
                ),
            ),
        ):
            output = patched_forward(
                _FakeEncoder(),
                input_features,
                feature_lens,
                aftercnn_lens,
            )

        self.assertTrue(torch.equal(output, torch.full_like(output, 3)))
        self.assertEqual(
            [event[0] for event in events],
            ["frontend", "metadata_h2d", "max_seqlen", "transformer"],
        )
        self.assertEqual(
            events[0][1:],
            ((1, 1, 128, 2), (2,), (1, 0), (0, 1), (2,), (1,)),
        )
        self.assertEqual(events[-1][1:], ((1, 1024), (0, 1), 1, (0, 1)))

    def test_dispatch_uses_combined_installer_without_partial_fallback(self) -> None:
        with (
            patch.object(
                combined_patch,
                "audio_frontend_cudagraph_enabled",
                return_value=True,
            ),
            patch.object(
                combined_patch,
                "audio_transformer_cudagraph_enabled",
                return_value=True,
            ),
            patch.object(
                combined_patch,
                "install_audio_encoder_cudagraph_patch",
                return_value=False,
            ) as install_combined,
            patch.object(
                combined_patch,
                "install_audio_frontend_cudagraph_patch",
            ) as install_frontend,
            patch.object(
                combined_patch,
                "install_audio_transformer_cudagraph_patch",
            ) as install_transformer,
            patch.object(
                combined_patch,
                "install_audio_cpu_metadata_pack_patch",
            ) as install_metadata,
        ):
            self.assertFalse(
                combined_patch.install_requested_audio_cudagraph_patches()
            )

        install_combined.assert_called_once_with()
        install_frontend.assert_not_called()
        install_transformer.assert_not_called()
        install_metadata.assert_not_called()

    def test_dispatch_preserves_each_single_runner_path(self) -> None:
        cases = (
            (True, False, "frontend"),
            (False, True, "transformer"),
        )
        for frontend_enabled, transformer_enabled, expected in cases:
            with self.subTest(expected=expected):
                installers = {
                    "frontend": Mock(return_value=True),
                    "transformer": Mock(return_value=True),
                }
                with (
                    patch.object(
                        combined_patch,
                        "audio_frontend_cudagraph_enabled",
                        return_value=frontend_enabled,
                    ),
                    patch.object(
                        combined_patch,
                        "audio_transformer_cudagraph_enabled",
                        return_value=transformer_enabled,
                    ),
                    patch.object(
                        combined_patch,
                        "install_audio_frontend_cudagraph_patch",
                        installers["frontend"],
                    ),
                    patch.object(
                        combined_patch,
                        "install_audio_transformer_cudagraph_patch",
                        installers["transformer"],
                    ),
                ):
                    self.assertTrue(
                        combined_patch.install_requested_audio_cudagraph_patches()
                    )

                installers[expected].assert_called_once_with()
                other = "transformer" if expected == "frontend" else "frontend"
                installers[other].assert_not_called()

    def test_combined_install_is_idempotent_for_both_single_installers(self) -> None:
        from vllm.model_executor.models import qwen3_asr
        from vllm.model_executor.models.qwen3_asr import (
            Qwen3OmniMoeAudioEncoder,
        )

        def clean_forward(self):
            del self

        def clean_field_config(inputs):
            return inputs

        install_calls = []

        def install_metadata(*, frontend_runner, transformer_runner):
            install_calls.append((frontend_runner, transformer_runner))

            def installed_forward(self):
                del self

            def installed_field_config(inputs):
                return inputs

            setattr(installed_forward, METADATA_PATCH_MARKER, True)
            setattr(installed_field_config, METADATA_PATCH_MARKER, True)
            Qwen3OmniMoeAudioEncoder.forward = installed_forward
            qwen3_asr._qwen3asr_field_config = installed_field_config
            return True

        with (
            patch.dict(os.environ, _combined_environment(), clear=True),
            patch.object(
                Qwen3OmniMoeAudioEncoder,
                "forward",
                clean_forward,
            ),
            patch.object(
                qwen3_asr,
                "_qwen3asr_field_config",
                clean_field_config,
            ),
            patch.object(
                combined_patch,
                "install_audio_cpu_metadata_pack_patch",
                side_effect=install_metadata,
            ),
        ):
            self.assertTrue(
                combined_patch.install_audio_encoder_cudagraph_patch()
            )
            installed_forward = Qwen3OmniMoeAudioEncoder.forward
            self.assertTrue(getattr(installed_forward, FRONTEND_PATCH_MARKER))
            self.assertTrue(getattr(installed_forward, TRANSFORMER_PATCH_MARKER))
            self.assertTrue(
                getattr(installed_forward, combined_patch._PATCH_MARKER)
            )
            self.assertTrue(
                combined_patch.install_audio_encoder_cudagraph_patch()
            )
            self.assertTrue(install_audio_frontend_cudagraph_patch())
            self.assertTrue(install_audio_transformer_cudagraph_patch())

        self.assertEqual(
            install_calls,
            [
                (
                    combined_patch.run_audio_frontend_cudagraph,
                    combined_patch.run_audio_transformer_cudagraph,
                )
            ],
        )

    def test_combined_install_refuses_metadata_only_forward(self) -> None:
        from vllm.model_executor.models import qwen3_asr
        from vllm.model_executor.models.qwen3_asr import (
            Qwen3OmniMoeAudioEncoder,
        )

        def metadata_forward(self):
            del self

        def metadata_field_config(inputs):
            return inputs

        setattr(metadata_forward, METADATA_PATCH_MARKER, True)
        setattr(metadata_field_config, METADATA_PATCH_MARKER, True)

        with (
            patch.dict(os.environ, _combined_environment(), clear=True),
            patch.object(
                Qwen3OmniMoeAudioEncoder,
                "forward",
                metadata_forward,
            ),
            patch.object(
                qwen3_asr,
                "_qwen3asr_field_config",
                metadata_field_config,
            ),
            patch.object(
                combined_patch,
                "install_audio_cpu_metadata_pack_patch",
            ) as install_metadata,
        ):
            self.assertFalse(
                combined_patch.install_audio_encoder_cudagraph_patch()
            )

        install_metadata.assert_not_called()

    def test_combined_install_refuses_each_single_runner_forward(self) -> None:
        from vllm.model_executor.models import qwen3_asr
        from vllm.model_executor.models.qwen3_asr import (
            Qwen3OmniMoeAudioEncoder,
        )

        for marker in (FRONTEND_PATCH_MARKER, TRANSFORMER_PATCH_MARKER):
            with self.subTest(marker=marker):
                def single_runner_forward(self):
                    del self

                def metadata_field_config(inputs):
                    return inputs

                setattr(single_runner_forward, METADATA_PATCH_MARKER, True)
                setattr(single_runner_forward, marker, True)
                setattr(metadata_field_config, METADATA_PATCH_MARKER, True)

                with (
                    patch.dict(
                        os.environ,
                        _combined_environment(),
                        clear=True,
                    ),
                    patch.object(
                        Qwen3OmniMoeAudioEncoder,
                        "forward",
                        single_runner_forward,
                    ),
                    patch.object(
                        qwen3_asr,
                        "_qwen3asr_field_config",
                        metadata_field_config,
                    ),
                    patch.object(
                        combined_patch,
                        "install_audio_cpu_metadata_pack_patch",
                    ) as install_metadata,
                ):
                    self.assertFalse(
                        combined_patch.install_audio_encoder_cudagraph_patch()
                    )

                install_metadata.assert_not_called()

    def test_combined_marker_without_prerequisite_markers_raises(self) -> None:
        from vllm.model_executor.models import qwen3_asr
        from vllm.model_executor.models.qwen3_asr import (
            Qwen3OmniMoeAudioEncoder,
        )

        def partial_forward(self):
            del self

        def clean_field_config(inputs):
            return inputs

        setattr(partial_forward, combined_patch._PATCH_MARKER, True)
        with (
            patch.dict(os.environ, _combined_environment(), clear=True),
            patch.object(
                Qwen3OmniMoeAudioEncoder,
                "forward",
                partial_forward,
            ),
            patch.object(
                qwen3_asr,
                "_qwen3asr_field_config",
                clean_field_config,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "partially installed"):
                combined_patch.install_audio_encoder_cudagraph_patch()


if __name__ == "__main__":
    unittest.main()
