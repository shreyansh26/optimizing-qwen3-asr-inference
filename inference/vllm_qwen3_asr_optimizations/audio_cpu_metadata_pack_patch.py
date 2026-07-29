"""Keep Qwen3-ASR audio lengths on CPU and pack valid rows in Triton."""

from __future__ import annotations

from functools import wraps
from importlib import metadata as importlib_metadata
from itertools import accumulate
import json
import os
from pathlib import Path
import tomllib
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from audio_cpu_maxseqlen_patch import (
    _is_supported_encoder,
    install_audio_cpu_maxseqlen_patch,
)
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton


logger = init_logger("vllm.qwen3_asr_audio_cpu_metadata")

ENV_NAME = "ASR_AUDIO_CPU_METADATA_PACK"
MAX_SEQLEN_ENV_NAME = "ASR_AUDIO_CPU_MAXSEQLEN"
_PATCH_MARKER = "_asr_audio_cpu_metadata_pack_patch"

_EXPECTED_VLLM_VERSION = "0.24.0+cu129"
_EXPECTED_VLLM_WHEEL_HASH = (
    "sha256:597949743f2a00c0539d9e2ff0f67b32c608a378e973f99e7529fd6fa9445f70"
)

# Previous exact-source guard retained for reference. Compatibility is now
# selected by the installed vLLM version and locked wheel hash above.
# _EXPECTED_FORWARD_SHA256 = (
#     "d7b4e3db2c0157a16eda90e63bcad09a28506a8991ac1dffc18413059d793807"
# )
# _EXPECTED_FIELD_CONFIG_SHA256 = (
#     "8a17c0eb01cfbf9f427de7e9dc24fcc91216f2bda846c91cab0b4addca5a3cb9"
# )
# _EXPECTED_PROCESS_AUDIO_SHA256 = (
#     "dcda6b1cc3c3dbb62f77fdc6e30bd34c3f174e3622c346fa383bf19f75fe945d"
# )
# _EXPECTED_CNN_OUTPUT_LENGTHS_SHA256 = (
#     "70a15691200217bbea5e938c75b044e006e61c4401cbda8900cbda8ede307950"
# )
# _EXPECTED_AUDIO_OUTPUT_LENGTHS_SHA256 = (
#     "eb472e321ad393edee5329d76a78b8a78bd11d54b1cbda552a344c95adf5b839"
# )


@triton.jit
def _pack_valid_rows_kernel(
    padded_ptr,
    metadata_ptr,
    output_ptr,
    padded_batch_stride,
    padded_row_stride,
    padded_hidden_stride,
    num_chunks,
    padded_rows,
    hidden_size: tl.constexpr,
    block_hidden: tl.constexpr,
):
    """Copy one padded row per program into its CPU-computed output slot."""
    program = tl.program_id(0)
    chunk = program // padded_rows
    row = program % padded_rows
    columns = tl.arange(0, block_hidden)

    valid_rows = tl.load(metadata_ptr + chunk).to(tl.int64)
    output_offset = tl.load(metadata_ptr + num_chunks + chunk).to(tl.int64)
    valid = (row < valid_rows) & (columns < hidden_size)

    values = tl.load(
        padded_ptr
        + chunk * padded_batch_stride
        + row * padded_row_stride
        + columns * padded_hidden_stride,
        mask=valid,
    )
    tl.store(
        output_ptr + (output_offset + row) * hidden_size + columns,
        values,
        mask=valid,
    )


def audio_cpu_metadata_pack_enabled() -> bool:
    """Return the strict boolean environment gate for this experiment."""
    raw_value = os.environ.get(ENV_NAME, "0")
    if raw_value not in {"0", "1"}:
        raise ValueError(f"{ENV_NAME} must be 0 or 1, got {raw_value!r}")
    return raw_value == "1"


def _installed_vllm_wheel_is_supported(
    *,
    _lock_path: Path | None = None,
) -> bool:
    """Check the installed vLLM version and its locked wheel provenance."""
    try:
        distribution = importlib_metadata.distribution("vllm")
        if distribution.version != _EXPECTED_VLLM_VERSION:
            return False

        direct_url_text = distribution.read_text("direct_url.json")
        if direct_url_text is None:
            return False
        wheel_url = json.loads(direct_url_text).get("url")
        if not isinstance(wheel_url, str):
            return False

        lock_path = _lock_path or Path(__file__).resolve().parents[2] / "uv.lock"
        lock_data = tomllib.loads(lock_path.read_text())
    except (
        importlib_metadata.PackageNotFoundError,
        json.JSONDecodeError,
        OSError,
        tomllib.TOMLDecodeError,
    ):
        return False

    for package in lock_data.get("package", []):
        if (
            package.get("name") != "vllm"
            or package.get("version") != _EXPECTED_VLLM_VERSION
        ):
            continue
        return any(
            wheel.get("url") == wheel_url
            and wheel.get("hash") == _EXPECTED_VLLM_WHEEL_HASH
            for wheel in package.get("wheels", [])
        )
    return False


def _expected_audio_output_lengths(feature_lens: torch.Tensor) -> torch.Tensor:
    """Mirror Qwen3-ASR's three-convolution whole-audio length formula."""
    input_lengths_leave = feature_lens % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return (
        ((feat_lengths - 1) // 2 + 1 - 1) // 2
        + 1
        + (feature_lens // 100) * 13
    )


def _cpu_input_is_supported(
    encoder: Any,
    input_features: torch.Tensor,
    feature_lens: torch.Tensor,
    aftercnn_lens: torch.Tensor,
    *,
    model_cls: type[Any],
    flash_backend: Any,
) -> bool:
    if type(encoder) is not model_cls or not hasattr(encoder, "attn_backend"):
        return False
    if not _is_supported_encoder(
        encoder,
        model_cls=model_cls,
        flash_backend=flash_backend,
    ):
        return False
    if encoder.training or torch.is_grad_enabled():
        return False
    if (
        not isinstance(input_features, torch.Tensor)
        or input_features.ndim != 2
        or not input_features.is_cuda
        or input_features.dtype != torch.bfloat16
        or input_features.shape[0] != 128
    ):
        return False
    for lengths in (feature_lens, aftercnn_lens):
        if (
            not isinstance(lengths, torch.Tensor)
            or lengths.device.type != "cpu"
            or lengths.ndim != 1
            or lengths.dtype not in {torch.int32, torch.int64}
            or not lengths.is_contiguous()
        ):
            return False
    if feature_lens.shape != aftercnn_lens.shape or feature_lens.numel() == 0:
        return False
    if bool((feature_lens <= 0).any()):
        return False
    if int(feature_lens.sum()) != input_features.shape[1]:
        return False
    expected_aftercnn = _expected_audio_output_lengths(feature_lens)
    return torch.equal(expected_aftercnn, aftercnn_lens)


def _build_cpu_metadata(
    feature_lens: torch.Tensor,
    aftercnn_lens: torch.Tensor,
    *,
    n_window: int,
    n_window_infer: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Build pack lengths/offsets and attention offsets without a CUDA readback."""
    raw_chunk_size = n_window * 2
    chunk_num = torch.ceil(feature_lens / raw_chunk_size).long()
    chunk_lengths = torch.full(
        (int(chunk_num.sum()),),
        raw_chunk_size,
        dtype=torch.long,
        device="cpu",
    )
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    # The public guard accepts both int32 and int64 CPU length tensors, while
    # chunk_lengths is intentionally long for split()/indexing consumers.
    # Normalize the tail values before indexed assignment so int32 inputs use
    # the same supported fast path instead of failing on a dtype mismatch.
    chunk_lengths[tail_chunk_index] = (
        feature_lens % raw_chunk_size
    ).to(chunk_lengths.dtype)
    chunk_lengths[chunk_lengths == 0] = raw_chunk_size

    pack_lengths = chunk_lengths
    for _ in range(3):
        pack_lengths = (pack_lengths - 1) // 2 + 1
    pack_length_values = [int(value) for value in pack_lengths.tolist()]
    pack_offsets = [0, *accumulate(pack_length_values)]

    max_pack_length = max(pack_length_values)
    window_aftercnn = max_pack_length * (n_window_infer // raw_chunk_size)
    attention_lengths: list[int] = []
    for cnn_len in aftercnn_lens.tolist():
        cnn_len = int(cnn_len)
        num_full_chunks, remainder = divmod(cnn_len, window_aftercnn)
        attention_lengths.extend([window_aftercnn] * num_full_chunks)
        if remainder:
            attention_lengths.append(remainder)
    cu_seqlens = [0, *accumulate(attention_lengths)]

    if pack_offsets[-1] != int(aftercnn_lens.sum()):
        raise ValueError("CPU pack metadata does not match whole-audio lengths")
    if cu_seqlens[-1] != pack_offsets[-1]:
        raise ValueError("CPU attention metadata does not match packed rows")

    # The first half is per-chunk valid rows; the second half is per-chunk
    # output offsets.  The final cumulative offset is only needed on the CPU.
    pack_metadata = torch.tensor(
        pack_length_values + pack_offsets[:-1],
        dtype=torch.int32,
        device="cpu",
    )
    return chunk_lengths, pack_metadata, cu_seqlens


def pack_valid_rows(
    padded: torch.Tensor,
    pack_metadata_cpu: torch.Tensor,
    *,
    async_tensor_h2d: Any,
) -> torch.Tensor:
    """Pack `[chunk, row, hidden]` using CPU-known lengths and offsets."""
    if (
        padded.ndim != 3
        or not padded.is_cuda
        or padded.dtype != torch.bfloat16
        or padded.shape[2] != 1024
        or pack_metadata_cpu.device.type != "cpu"
        or pack_metadata_cpu.dtype != torch.int32
        or pack_metadata_cpu.ndim != 1
        or pack_metadata_cpu.numel() != padded.shape[0] * 2
    ):
        raise ValueError("Unsupported tensor layout for audio valid-row pack")

    num_chunks, padded_rows, hidden_size = padded.shape
    metadata_values = pack_metadata_cpu.tolist()
    lengths = metadata_values[:num_chunks]
    offsets = metadata_values[num_chunks:]
    running_offset = 0
    for length, offset in zip(lengths, offsets, strict=True):
        if length <= 0 or length > padded_rows:
            raise ValueError("Invalid CPU row lengths for audio valid-row pack")
        if offset != running_offset:
            raise ValueError("Invalid CPU row offsets for audio valid-row pack")
        running_offset += length

    total_rows = running_offset
    output = torch.empty(
        (total_rows, hidden_size),
        device=padded.device,
        dtype=padded.dtype,
    )
    metadata = async_tensor_h2d(
        pack_metadata_cpu,
        dtype=torch.int32,
        device=padded.device,
    )
    block_hidden = triton.next_power_of_2(hidden_size)
    _pack_valid_rows_kernel[(num_chunks * padded_rows,)](
        padded,
        metadata,
        output,
        padded.stride(0),
        padded.stride(1),
        padded.stride(2),
        num_chunks,
        padded_rows,
        hidden_size,
        block_hidden,
        num_warps=8,
    )
    return output


def _lengths_on_device(
    lengths: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if lengths.device == device:
        return lengths
    return lengths.to(device=device, non_blocking=True)


def run_audio_frontend_eager(
    encoder: Any,
    padded_feature: torch.Tensor,
    chunk_lengths_cpu: torch.Tensor,
    pack_metadata_cpu: torch.Tensor,
    cu_seqlens_values: tuple[int, ...],
    feature_lens_values: tuple[int, ...],
    aftercnn_lens_values: tuple[int, ...],
    *,
    async_tensor_h2d: Any,
) -> torch.Tensor:
    """Run the eager audio frontend and valid-row pack."""
    del chunk_lengths_cpu, cu_seqlens_values, feature_lens_values
    del aftercnn_lens_values
    if padded_feature.size(0) <= encoder.conv_chunksize:
        padded_embed = F.gelu(encoder.conv2d1(padded_feature))
        padded_embed = F.gelu(encoder.conv2d2(padded_embed))
        padded_embed = F.gelu(encoder.conv2d3(padded_embed))
    else:
        padded_embeds = []
        for chunk in padded_feature.split(encoder.conv_chunksize, dim=0):
            padded_embed = F.gelu(encoder.conv2d1(chunk))
            padded_embed = F.gelu(encoder.conv2d2(padded_embed))
            padded_embed = F.gelu(encoder.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)

    batch, channels, frequency, time = padded_embed.size()
    padded_embed = encoder.conv_out(
        padded_embed.permute(0, 3, 1, 2)
        .contiguous()
        .view(batch, time, channels * frequency)
    )
    positional_embedding = (
        encoder.positional_embedding.positional_embedding[
            : padded_embed.shape[1], :
        ]
        .unsqueeze(0)
        .to(padded_embed.dtype)
    )
    padded_embed = padded_embed + positional_embedding

    if padded_embed.is_cuda:
        return pack_valid_rows(
            padded_embed,
            pack_metadata_cpu,
            async_tensor_h2d=async_tensor_h2d,
        )

    num_chunks = padded_embed.shape[0]
    metadata_values = [int(value) for value in pack_metadata_cpu.tolist()]
    lengths = metadata_values[:num_chunks]
    return torch.cat(
        [padded_embed[index, :length] for index, length in enumerate(lengths)],
        dim=0,
    )


def run_audio_transformer_eager(
    encoder: Any,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: torch.Tensor | None,
    *,
    cu_seqlens_values: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Run only the audio transformer and output projection."""
    del cu_seqlens_values
    for encoder_layer in encoder.layers:
        hidden_states = encoder_layer(
            hidden_states,
            cu_seqlens,
            max_seqlen,
        )

    hidden_states = encoder.ln_post(hidden_states)
    hidden_states = encoder.proj1(hidden_states)
    hidden_states = encoder.act(hidden_states)
    hidden_states = encoder.proj2(hidden_states)
    return hidden_states


def _make_patched_forward(
    original_forward: Any,
    *,
    model_cls: type[Any],
    flash_backend: Any,
    async_tensor_h2d: Any,
    frontend_runner: Any | None = None,
    transformer_runner: Any | None = None,
) -> Any:
    @wraps(original_forward)
    def patched_forward(
        self: Any,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
        aftercnn_lens: torch.Tensor,
    ) -> torch.Tensor:
        if not _cpu_input_is_supported(
            self,
            input_features,
            feature_lens,
            aftercnn_lens,
            model_cls=model_cls,
            flash_backend=flash_backend,
        ):
            return original_forward(
                self,
                input_features,
                _lengths_on_device(feature_lens, input_features.device),
                _lengths_on_device(aftercnn_lens, input_features.device),
            )

        chunk_lengths, pack_metadata_cpu, cu_seqlens_cpu = _build_cpu_metadata(
            feature_lens,
            aftercnn_lens,
            n_window=self.n_window,
            n_window_infer=self.n_window_infer,
        )

        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(
            chunk_list,
            batch_first=True,
        ).transpose(1, 2)
        padded_feature = padded_feature.unsqueeze(1)

        selected_frontend_runner = frontend_runner or run_audio_frontend_eager
        hidden_states = selected_frontend_runner(
            self,
            padded_feature,
            chunk_lengths,
            pack_metadata_cpu,
            tuple(cu_seqlens_cpu),
            tuple(int(value) for value in feature_lens.tolist()),
            tuple(int(value) for value in aftercnn_lens.tolist()),
            async_tensor_h2d=async_tensor_h2d,
        )
        cu_seqlens = async_tensor_h2d(
            cu_seqlens_cpu,
            dtype=torch.int32,
            device=input_features.device,
        )
        max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens)

        selected_transformer_runner = transformer_runner or run_audio_transformer_eager
        return selected_transformer_runner(
            self,
            hidden_states,
            cu_seqlens,
            max_seqlen,
            cu_seqlens_values=tuple(cu_seqlens_cpu),
        )

    setattr(patched_forward, _PATCH_MARKER, True)
    return patched_forward


def _make_cpu_field_config(
    original_field_config: Any,
    field_config_cls: type[Any],
) -> Any:
    @wraps(original_field_config)
    def patched_field_config(
        hf_inputs: Mapping[str, torch.Tensor],
    ) -> Mapping[str, Any]:
        configs = dict(original_field_config(hf_inputs))
        configs["audio_feature_lengths"] = field_config_cls.batched(
            "audio",
            keep_on_cpu=True,
        )
        return configs

    setattr(patched_field_config, _PATCH_MARKER, True)
    return patched_field_config


def install_audio_cpu_metadata_pack_patch(
    *,
    frontend_runner: Any | None = None,
    transformer_runner: Any | None = None,
) -> bool:
    """Install the exact-version CPU-metadata audio encoder fast path."""
    if not audio_cpu_metadata_pack_enabled():
        return False
    if os.environ.get(MAX_SEQLEN_ENV_NAME, "0") != "1":
        logger.warning(
            "%s requires %s=1; leaving the installed model unchanged",
            ENV_NAME,
            MAX_SEQLEN_ENV_NAME,
        )
        return False
    if not _installed_vllm_wheel_is_supported():
        logger.warning(
            "Installed vLLM version or wheel hash does not match the CPU "
            "metadata patch; leaving it unchanged"
        )
        return False
    from vllm.model_executor.models import qwen3_asr as asr_module
    from vllm.model_executor.models.qwen3_asr import Qwen3OmniMoeAudioEncoder
    from vllm.multimodal.inputs import MultiModalFieldConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    model_cls = Qwen3OmniMoeAudioEncoder
    current_forward = model_cls.forward
    current_field_config = asr_module._qwen3asr_field_config
    forward_patched = bool(getattr(current_forward, _PATCH_MARKER, False))
    field_patched = bool(getattr(current_field_config, _PATCH_MARKER, False))
    if forward_patched or field_patched:
        if forward_patched and field_patched:
            return True
        raise RuntimeError("Audio CPU metadata patch is only partially installed")

    async_tensor_h2d = current_forward.__globals__.get("async_tensor_h2d")
    if async_tensor_h2d is None:
        logger.warning("Qwen3-ASR forward has no async_tensor_h2d seam")
        return False

    patched_forward = _make_patched_forward(
        current_forward,
        model_cls=model_cls,
        flash_backend=AttentionBackendEnum.FLASH_ATTN,
        async_tensor_h2d=async_tensor_h2d,
        frontend_runner=frontend_runner,
        transformer_runner=transformer_runner,
    )
    patched_field_config = _make_cpu_field_config(
        current_field_config,
        MultiModalFieldConfig,
    )

    # Complete the compatibility check and wrapper construction before
    # installing even the max-seqlen prerequisite. A mismatch therefore leaves
    # all three global seams unchanged.
    if not install_audio_cpu_maxseqlen_patch(
        model_cls=model_cls,
        flash_backend=AttentionBackendEnum.FLASH_ATTN,
    ):
        logger.warning("Audio CPU max-seqlen prerequisite was not installed")
        return False

    model_cls.forward = patched_forward
    asr_module._qwen3asr_field_config = patched_field_config
    return True
