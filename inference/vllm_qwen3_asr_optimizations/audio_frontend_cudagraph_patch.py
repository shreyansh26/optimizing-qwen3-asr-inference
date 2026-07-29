"""Exact-shape CUDA graph cache for the Qwen3-ASR audio frontend."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import os
import threading
from typing import Any, Callable

import torch
import torch.nn.functional as F

from audio_cpu_metadata_pack_patch import (
    ENV_NAME as METADATA_ENV_NAME,
    MAX_SEQLEN_ENV_NAME,
    _PATCH_MARKER as METADATA_PATCH_MARKER,
    install_audio_cpu_metadata_pack_patch,
    run_audio_frontend_eager,
)
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton


logger = init_logger("vllm.qwen3_asr_audio_frontend_cudagraph")

ENV_NAME = "ASR_AUDIO_FRONTEND_CUDAGRAPH"
_PATCH_MARKER = "_asr_audio_frontend_cudagraph_patch"
_CACHE_ATTR = "_asr_audio_frontend_cudagraph_cache"
_CACHE_CREATION_LOCK = threading.Lock()

# Exact Qwen3-ASR frontend geometry:
#
#   padded_feature: [chunks, 1 channel, 128 mel bins, 100 frames]
#   three stride-2 convolutions:
#       100 -> 50 -> 25 -> 13 time rows
#   conv_out:       [chunks, 13, 1024]
#   packed output:  [sum(valid rows), 1024]
#
# A 29--30 second natural request contains 29 or 30 raw chunks.  Full chunks
# contribute 13 rows, so admitted packed row counts span 29*13=377 through
# 30*13=390. These calculations are kept executable below.
_INPUT_CHANNELS = 1
_INPUT_MELS = 128
_INPUT_FRAMES = 100
_PACKED_WIDTH = 1024
_CONV_DOWNSAMPLE_STAGES = 3
_CONV_DOWNSAMPLE_FACTOR = 1 << _CONV_DOWNSAMPLE_STAGES
_CONV_OUTPUT_ROWS = (
    _INPUT_FRAMES + _CONV_DOWNSAMPLE_FACTOR - 1
) // _CONV_DOWNSAMPLE_FACTOR
_MIN_NATURAL_CHUNKS = 29
_MAX_NATURAL_CHUNKS = 30
_TAIL_PACKED_ROWS = frozenset()
_NATURAL_PACKED_ROWS = frozenset(
    range(
        _MIN_NATURAL_CHUNKS * _CONV_OUTPUT_ROWS,
        _MAX_NATURAL_CHUNKS * _CONV_OUTPUT_ROWS + 1,
    )
)
_SUPPORTED_PACKED_ROWS = _NATURAL_PACKED_ROWS
_SUPPORTED_CHUNKS = frozenset(
    range(_MIN_NATURAL_CHUNKS, _MAX_NATURAL_CHUNKS + 1)
)

# Admission policy for the benchmark's nominal 30-second audio family.  Mel
# frames are 10 ms apart: 2897--3000 frames corresponds to 28.97--30.00 s.
_NATURAL_FEATURE_LENGTH_MIN = _MIN_NATURAL_CHUNKS * _INPUT_FRAMES - 3
_NATURAL_FEATURE_LENGTH_MAX = _MAX_NATURAL_CHUNKS * _INPUT_FRAMES
_BASELINE_PACKED_ROWS = _MIN_NATURAL_CHUNKS * _CONV_OUTPUT_ROWS
_ROW_TO_MAX_FEATURE_BIAS = (
    _MIN_NATURAL_CHUNKS * _INPUT_FRAMES
    - _BASELINE_PACKED_ROWS * _CONV_DOWNSAMPLE_FACTOR
)
_ROW_TO_MIN_FEATURE_BIAS = (
    _ROW_TO_MAX_FEATURE_BIAS - _CONV_DOWNSAMPLE_FACTOR + 1
)

# Attention groups at most eight 100-frame chunks. Each contributes 13 rows:
# 8 * 13 = 104 rows per attention sequence.
_CHUNKS_PER_ATTENTION_WINDOW = 8
_ATTENTION_WINDOW_ROWS = _CHUNKS_PER_ATTENTION_WINDOW * _CONV_OUTPUT_ROWS
_MAX_CACHE_ENTRIES = len(_SUPPORTED_PACKED_ROWS)
_MAX_PROBATION_KEYS = (
    _NATURAL_FEATURE_LENGTH_MAX - _NATURAL_FEATURE_LENGTH_MIN + 1
)
_WARMUP_ITERATIONS = 3
_PROBATION_OBSERVATIONS = 8


@triton.jit
def _pack_valid_rows_device_kernel(
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
    """Pack valid [chunk, row, hidden] data using device-resident metadata.

    metadata_ptr stores [valid_rows_per_chunk | output_offset_per_chunk], each
    section having num_chunks int32 entries. Grid = num_chunks * padded_rows:
    one program owns one candidate row and vectorizes over block_hidden columns.
    """
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


@dataclass(frozen=True)
class FrontendGraphKey:
    """All static launch and CPU-metadata state for one frontend graph."""

    padded_shape: tuple[int, int, int, int]
    padded_stride: tuple[int, int, int, int]
    packed_rows: int
    dtype: torch.dtype
    device_type: str
    device_index: int | None
    chunk_lengths_values: tuple[int, ...]
    pack_metadata_values: tuple[int, ...]
    cu_seqlens_values: tuple[int, ...]
    feature_lens_values: tuple[int, ...]
    aftercnn_lens_values: tuple[int, ...]


@dataclass(frozen=True)
class FrontendGraphSignature:
    """State consumed by the captured frontend, after exact-key admission."""

    padded_shape: tuple[int, int, int, int]
    padded_stride: tuple[int, int, int, int]
    packed_rows: int
    dtype: torch.dtype
    device_type: str
    device_index: int | None
    pack_metadata_values: tuple[int, ...]


@dataclass
class _FrontendGraphEntry:
    signature: FrontendGraphSignature
    static_padded_feature: torch.Tensor
    static_pack_metadata: torch.Tensor
    graph: Any
    output: torch.Tensor
    execution_stream: Any
    done_event: Any


def audio_frontend_cudagraph_enabled() -> bool:
    """Return the strict boolean environment gate for this experiment."""
    raw_value = os.environ.get(ENV_NAME, "0")
    if raw_value not in {"0", "1"}:
        raise ValueError(f"{ENV_NAME} must be 0 or 1, got {raw_value!r}")
    return raw_value == "1"


def _metadata_tuple(values: torch.Tensor | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(values, tuple):
        return tuple(int(value) for value in values)
    if values.device.type != "cpu" or values.ndim != 1:
        raise ValueError("Audio frontend metadata must be one-dimensional CPU data")
    return tuple(int(value) for value in values.tolist())


def _validate_pack_metadata_values(
    pack_metadata_values: tuple[int, ...],
    *,
    num_chunks: int,
) -> int | None:
    if len(pack_metadata_values) != num_chunks * 2:
        return None
    lengths = pack_metadata_values[:num_chunks]
    offsets = pack_metadata_values[num_chunks:]
    running_offset = 0
    for length, offset in zip(lengths, offsets, strict=True):
        if length <= 0 or length > _CONV_OUTPUT_ROWS or offset != running_offset:
            return None
        running_offset += length
    return running_offset


def _natural_feature_lengths_for_rows(packed_rows: int) -> tuple[int, ...]:
    """Return every single-audio feature length for one admitted natural row."""
    if packed_rows not in _NATURAL_PACKED_ROWS:
        return ()
    # For the 29-full-chunk baseline, row 377 ends at frame 2900. Every next
    # packed row represents the next group of up to eight input frames because
    # three stride-2 convolutions downsample by 2**3=8.
    first = max(
        _NATURAL_FEATURE_LENGTH_MIN,
        packed_rows * _CONV_DOWNSAMPLE_FACTOR + _ROW_TO_MIN_FEATURE_BIAS,
    )
    last = min(
        _NATURAL_FEATURE_LENGTH_MAX,
        packed_rows * _CONV_DOWNSAMPLE_FACTOR + _ROW_TO_MAX_FEATURE_BIAS,
    )
    return tuple(range(first, last + 1))


def _canonical_single_audio_natural_metadata(
    feature_length: int,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
] | None:
    """Derive the exact CPU metadata emitted for one 29--30 s audio."""
    if not (
        _NATURAL_FEATURE_LENGTH_MIN
        <= feature_length
        <= _NATURAL_FEATURE_LENGTH_MAX
    ):
        return None

    full_chunks, tail_frames = divmod(feature_length, _INPUT_FRAMES)
    chunk_lengths = (_INPUT_FRAMES,) * full_chunks
    if tail_frames:
        chunk_lengths += (tail_frames,)
    pack_lengths = tuple((length + 7) // 8 for length in chunk_lengths)
    offsets = [0]
    for length in pack_lengths:
        offsets.append(offsets[-1] + length)
    packed_rows = offsets[-1]
    if packed_rows not in _NATURAL_PACKED_ROWS:
        return None

    return (
        chunk_lengths,
        pack_lengths + tuple(offsets[:-1]),
        (*range(0, packed_rows, _ATTENTION_WINDOW_ROWS), packed_rows),
        (feature_length,),
        (packed_rows,),
    )


def _is_canonical_single_audio_natural_key(
    *,
    padded_shape: tuple[int, int, int, int],
    packed_rows: int,
    chunk_lengths_values: tuple[int, ...],
    pack_metadata_values: tuple[int, ...],
    cu_seqlens_values: tuple[int, ...],
    feature_lens_values: tuple[int, ...],
    aftercnn_lens_values: tuple[int, ...],
) -> bool:
    if len(feature_lens_values) != 1:
        return False
    expected = _canonical_single_audio_natural_metadata(
        feature_lens_values[0]
    )
    if expected is None:
        return False
    (
        expected_chunk_lengths,
        expected_pack_metadata,
        expected_cu_seqlens,
        expected_feature_lens,
        expected_aftercnn_lens,
    ) = expected
    return (
        packed_rows == expected_aftercnn_lens[0]
        and padded_shape
        == (
            len(expected_chunk_lengths),
            _INPUT_CHANNELS,
            _INPUT_MELS,
            _INPUT_FRAMES,
        )
        and chunk_lengths_values == expected_chunk_lengths
        and pack_metadata_values == expected_pack_metadata
        and cu_seqlens_values == expected_cu_seqlens
        and feature_lens_values == expected_feature_lens
        and aftercnn_lens_values == expected_aftercnn_lens
    )


def _make_frontend_graph_key(
    padded_feature: torch.Tensor,
    chunk_lengths_cpu: torch.Tensor,
    pack_metadata_cpu: torch.Tensor,
    cu_seqlens_values: tuple[int, ...],
    feature_lens_values: tuple[int, ...],
    aftercnn_lens_values: tuple[int, ...],
) -> FrontendGraphKey | None:
    if (
        not isinstance(padded_feature, torch.Tensor)
        or padded_feature.ndim != 4
        or padded_feature.shape[1:] != (_INPUT_CHANNELS, _INPUT_MELS, _INPUT_FRAMES)
        or padded_feature.dtype != torch.bfloat16
        or not isinstance(chunk_lengths_cpu, torch.Tensor)
        or chunk_lengths_cpu.device.type != "cpu"
        or chunk_lengths_cpu.ndim != 1
        or chunk_lengths_cpu.dtype not in {torch.int32, torch.int64}
        or not isinstance(pack_metadata_cpu, torch.Tensor)
        or pack_metadata_cpu.device.type != "cpu"
        or pack_metadata_cpu.ndim != 1
        or pack_metadata_cpu.dtype != torch.int32
    ):
        return None

    num_chunks = int(padded_feature.shape[0])
    chunk_lengths_values = _metadata_tuple(chunk_lengths_cpu)
    pack_metadata_values = _metadata_tuple(pack_metadata_cpu)
    packed_rows = _validate_pack_metadata_values(
        pack_metadata_values,
        num_chunks=num_chunks,
    )
    padded_shape = tuple(int(value) for value in padded_feature.shape)
    natural_key = _is_canonical_single_audio_natural_key(
        padded_shape=padded_shape,
        packed_rows=packed_rows if packed_rows is not None else -1,
        chunk_lengths_values=chunk_lengths_values,
        pack_metadata_values=pack_metadata_values,
        cu_seqlens_values=tuple(int(value) for value in cu_seqlens_values),
        feature_lens_values=tuple(int(value) for value in feature_lens_values),
        aftercnn_lens_values=tuple(int(value) for value in aftercnn_lens_values),
    )
    if (
        packed_rows is None
        or num_chunks not in _SUPPORTED_CHUNKS
        or packed_rows not in _SUPPORTED_PACKED_ROWS
        or not natural_key
        or len(chunk_lengths_values) != num_chunks
        or any(length <= 0 or length > _INPUT_FRAMES for length in chunk_lengths_values)
        or not cu_seqlens_values
        or cu_seqlens_values[0] != 0
        or cu_seqlens_values[-1] != packed_rows
        or any(
            left >= right
            for left, right in zip(cu_seqlens_values, cu_seqlens_values[1:])
        )
        or sum(aftercnn_lens_values) != packed_rows
        or len(feature_lens_values) != len(aftercnn_lens_values)
    ):
        return None

    return FrontendGraphKey(
        padded_shape=padded_shape,
        padded_stride=tuple(int(value) for value in padded_feature.stride()),
        packed_rows=packed_rows,
        dtype=padded_feature.dtype,
        device_type=padded_feature.device.type,
        device_index=padded_feature.device.index,
        chunk_lengths_values=chunk_lengths_values,
        pack_metadata_values=pack_metadata_values,
        cu_seqlens_values=tuple(int(value) for value in cu_seqlens_values),
        feature_lens_values=tuple(int(value) for value in feature_lens_values),
        aftercnn_lens_values=tuple(int(value) for value in aftercnn_lens_values),
    )


def _make_frontend_graph_signature(key: FrontendGraphKey) -> FrontendGraphSignature:
    """Reduce a fully admitted runtime key to captured-function inputs only."""
    return FrontendGraphSignature(
        padded_shape=key.padded_shape,
        padded_stride=key.padded_stride,
        packed_rows=key.packed_rows,
        dtype=key.dtype,
        device_type=key.device_type,
        device_index=key.device_index,
        pack_metadata_values=key.pack_metadata_values,
    )


def _output_is_supported(
    output: Any,
    key: FrontendGraphKey | FrontendGraphSignature,
) -> bool:
    return (
        isinstance(output, torch.Tensor)
        and output.shape == (key.packed_rows, _PACKED_WIDTH)
        and output.dtype == key.dtype
        and output.device.type == key.device_type
        and output.device.index == key.device_index
    )


def pack_valid_rows_from_device_metadata(
    padded: torch.Tensor,
    metadata: torch.Tensor,
    *,
    total_rows: int,
) -> torch.Tensor:
    """Pack `[chunk, row, hidden]` using graph-static device metadata."""
    if (
        padded.ndim != 3
        or padded.dtype != torch.bfloat16
        or padded.shape[1:] != (_CONV_OUTPUT_ROWS, _PACKED_WIDTH)
        or metadata.ndim != 1
        or metadata.dtype != torch.int32
        or metadata.numel() != padded.shape[0] * 2
        or metadata.device != padded.device
    ):
        raise ValueError("Unsupported tensor layout for graph audio row pack")

    num_chunks, padded_rows, hidden_size = padded.shape
    if padded.is_cuda:
        output = torch.empty(
            (total_rows, hidden_size),
            device=padded.device,
            dtype=padded.dtype,
        )
        block_hidden = triton.next_power_of_2(hidden_size)
        grid = (num_chunks * padded_rows,)
        _pack_valid_rows_device_kernel[grid](
            padded,
            metadata,
            output,
            # For contiguous [chunks, 13, 1024], these live strides are
            # [13 * 1024 = 13312, 1024, 1].
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

    metadata_values = _metadata_tuple(metadata.cpu())
    lengths = metadata_values[:num_chunks]
    return torch.cat(
        [padded[index, :length] for index, length in enumerate(lengths)],
        dim=0,
    )


def run_audio_frontend_with_device_metadata(
    encoder: Any,
    padded_feature: torch.Tensor,
    pack_metadata: torch.Tensor,
    *,
    total_rows: int,
) -> torch.Tensor:
    """Run the exact frontend with already-materialized metadata."""
    padded_embed = F.gelu(encoder.conv2d1(padded_feature))
    padded_embed = F.gelu(encoder.conv2d2(padded_embed))
    padded_embed = F.gelu(encoder.conv2d3(padded_embed))

    # Conv output is [chunks, 480 channels, frequency, 13 time rows].
    # Move time next to batch and flatten channels*frequency for conv_out;
    # conv_out produces the model-width [chunks, 13, 1024] tensor.
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
    return pack_valid_rows_from_device_metadata(
        padded_embed,
        pack_metadata,
        total_rows=total_rows,
    )


class _TorchCudaGraphBackend:
    """Small CUDA API seam so cache behavior has CUDA-hidden fake tests."""

    def supports(self, padded_feature: torch.Tensor) -> bool:
        return (
            torch.cuda.is_available()
            and padded_feature.is_cuda
            and padded_feature.dtype == torch.bfloat16
        )

    def is_current_stream_capturing(self) -> bool:
        return torch.cuda.is_current_stream_capturing()

    def make_device_metadata(
        self,
        metadata_cpu: torch.Tensor,
        *,
        device: torch.device,
        async_tensor_h2d: Any,
    ) -> torch.Tensor:
        return async_tensor_h2d(metadata_cpu, dtype=torch.int32, device=device)

    def create_shared_capture_resources(
        self,
        device: torch.device,
    ) -> tuple[Any, Any]:
        with torch.cuda.device(device):
            execution_stream = torch.cuda.Stream(device=device)
            graph_pool = torch.cuda.graph_pool_handle()
        return execution_stream, graph_pool

    def capture(
        self,
        function: Callable[[], torch.Tensor],
        *,
        device: torch.device,
        warmup_iterations: int,
        execution_stream: Any,
        graph_pool: Any,
    ) -> tuple[Any, torch.Tensor, Any, Any]:
        caller_stream = torch.cuda.current_stream(device)
        execution_stream.wait_stream(caller_stream)
        with torch.cuda.stream(execution_stream):
            for _ in range(warmup_iterations):
                function()
        caller_stream.wait_stream(execution_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph,
            pool=graph_pool,
            stream=execution_stream,
            capture_error_mode="thread_local",
        ):
            output = function()
        done_event = torch.cuda.Event(blocking=False)
        done_event.record(execution_stream)
        caller_stream.wait_event(done_event)
        return graph, output, execution_stream, done_event

    def replay_context(self, replay_stream: Any) -> Any:
        return torch.cuda.stream(replay_stream)

    def wait_stream(self, replay_stream: Any, device: torch.device) -> None:
        replay_stream.wait_stream(torch.cuda.current_stream(device))

    def record_done(self, done_event: Any, replay_stream: Any) -> None:
        done_event.record(replay_stream)

    def wait_done(self, done_event: Any, device: torch.device) -> None:
        torch.cuda.current_stream(device).wait_event(done_event)

    def replay(self, graph: Any) -> None:
        graph.replay()

    def replay_and_clone(
        self,
        entry: _FrontendGraphEntry,
        padded_feature: torch.Tensor,
    ) -> torch.Tensor:
        caller_stream = torch.cuda.current_stream(padded_feature.device)
        execution_stream = entry.execution_stream

        execution_stream.wait_stream(caller_stream)
        with torch.cuda.stream(execution_stream):
            entry.static_padded_feature.copy_(
                padded_feature,
                non_blocking=True,
            )
            # The asynchronous copy may outlive the caller's Python reference.
            padded_feature.record_stream(execution_stream)
            entry.graph.replay()
            output = entry.output.clone()
            entry.done_event.record(execution_stream)

        caller_stream.wait_event(entry.done_event)
        # Keep clone storage live through its asynchronous caller-stream use.
        output.record_stream(caller_stream)
        return output

    def equal(self, left: torch.Tensor, right: torch.Tensor) -> bool:
        return torch.equal(left, right)


class ExactShapeAudioFrontendGraphCache:
    """Admit full metadata keys onto verified graph-static signatures."""

    def __init__(
        self,
        *,
        backend: Any | None = None,
        max_entries: int = _MAX_CACHE_ENTRIES,
        max_probation_keys: int = _MAX_PROBATION_KEYS,
        warmup_iterations: int = _WARMUP_ITERATIONS,
    ) -> None:
        if (
            max_entries <= 0
            or max_probation_keys <= 0
            or warmup_iterations <= 0
        ):
            raise ValueError("CUDA graph cache limits must be positive")
        self._backend = backend or _TorchCudaGraphBackend()
        self._max_entries = max_entries
        self._max_probation_keys = max_probation_keys
        self._warmup_iterations = warmup_iterations
        self._entries: dict[FrontendGraphSignature, _FrontendGraphEntry] = {}
        self._admitted_keys: set[FrontendGraphKey] = set()
        self._rejected_keys: set[FrontendGraphKey] = set()
        self._probation_counts: dict[FrontendGraphKey, int] = {}
        self._execution_stream: Any | None = None
        self._graph_pool: Any | None = None
        # Cache state remains locked through cold admission so two eighth
        # observations cannot capture or alias-admit the same full key twice.
        self._global_lock = threading.Lock()
        # Every signature graph shares one private pool and one stream. Capture,
        # alias admission replay, and hot replay must therefore be mutually
        # exclusive. Enqueueing each complete copy -> graph -> clone transaction
        # under this lock also preserves clone-before-next-input FIFO ordering.
        self._transaction_lock = threading.Lock()
        self._logged_capacity = False
        self._logged_probation_capacity = False
        self._logged_replay = False

    @property
    def entry_count(self) -> int:
        with self._global_lock:
            return len(self._entries)

    @property
    def admitted_key_count(self) -> int:
        with self._global_lock:
            return len(self._admitted_keys)

    @property
    def rejected_count(self) -> int:
        with self._global_lock:
            return len(self._rejected_keys)

    @property
    def probation_key_count(self) -> int:
        with self._global_lock:
            return len(self._probation_counts)

    def _observe_key_locked(self, key: FrontendGraphKey) -> int:
        observations = self._probation_counts.get(key)
        if observations is None:
            if len(self._probation_counts) >= self._max_probation_keys:
                if not self._logged_probation_capacity:
                    logger.warning(
                        "Audio frontend CUDA graph probation is full at %d "
                        "exact metadata keys; new keys stay eager",
                        self._max_probation_keys,
                    )
                    self._logged_probation_capacity = True
                return 0
            observations = 0
        observations += 1
        if observations >= _PROBATION_OBSERVATIONS:
            self._probation_counts.pop(key, None)
        else:
            self._probation_counts[key] = observations
        return observations

    def _eager(
        self,
        encoder: Any,
        padded_feature: torch.Tensor,
        chunk_lengths_cpu: torch.Tensor,
        pack_metadata_cpu: torch.Tensor,
        cu_seqlens_values: tuple[int, ...],
        feature_lens_values: tuple[int, ...],
        aftercnn_lens_values: tuple[int, ...],
        async_tensor_h2d: Any,
    ) -> torch.Tensor:
        return run_audio_frontend_eager(
            encoder,
            padded_feature,
            chunk_lengths_cpu,
            pack_metadata_cpu,
            cu_seqlens_values,
            feature_lens_values,
            aftercnn_lens_values,
            async_tensor_h2d=async_tensor_h2d,
        )

    def run(
        self,
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
        key = _make_frontend_graph_key(
            padded_feature,
            chunk_lengths_cpu,
            pack_metadata_cpu,
            cu_seqlens_values,
            feature_lens_values,
            aftercnn_lens_values,
        )
        if (
            key is None
            or bool(getattr(encoder, "training", True))
            or torch.is_grad_enabled()
            or padded_feature.size(0) > getattr(encoder, "conv_chunksize", 0)
            or not self._backend.supports(padded_feature)
            or self._backend.is_current_stream_capturing()
        ):
            return self._eager(
                encoder,
                padded_feature,
                chunk_lengths_cpu,
                pack_metadata_cpu,
                tuple(cu_seqlens_values),
                tuple(feature_lens_values),
                tuple(aftercnn_lens_values),
                async_tensor_h2d,
            )

        signature = _make_frontend_graph_signature(key)
        capture_output: torch.Tensor | None = None
        replay_entry: _FrontendGraphEntry | None = None
        with self._global_lock:
            if key not in self._rejected_keys:
                signature_entry = self._entries.get(signature)
                if key in self._admitted_keys:
                    replay_entry = signature_entry
                elif (
                    signature_entry is None
                    and len(self._entries) >= self._max_entries
                ):
                    if not self._logged_capacity:
                        logger.warning(
                            "Audio frontend CUDA graph cache is full at %d "
                            "signatures; using eager frontend for shape=%s",
                            self._max_entries,
                            key.padded_shape,
                        )
                        self._logged_capacity = True
                else:
                    observations = self._observe_key_locked(key)
                    if observations == _PROBATION_OBSERVATIONS:
                        if signature_entry is None:
                            _, capture_output = self._capture_entry_locked(
                                encoder,
                                padded_feature,
                                chunk_lengths_cpu,
                                pack_metadata_cpu,
                                key,
                                signature,
                                async_tensor_h2d,
                            )
                        else:
                            capture_output = self._admit_existing_entry_locked(
                                encoder,
                                padded_feature,
                                chunk_lengths_cpu,
                                pack_metadata_cpu,
                                key,
                                signature_entry,
                                async_tensor_h2d,
                            )
                    elif observations > 0:
                        logger.info(
                            "Audio frontend CUDA graph probation uses eager "
                            "frontend shape=%s rows=%d feature_lens=%s "
                            "observation=%d/%d signatures=%d/%d",
                            key.padded_shape,
                            key.packed_rows,
                            key.feature_lens_values,
                            observations,
                            _PROBATION_OBSERVATIONS,
                            len(self._entries),
                            self._max_entries,
                        )

        if capture_output is not None:
            return capture_output
        if replay_entry is None:
            return self._eager(
                encoder,
                padded_feature,
                chunk_lengths_cpu,
                pack_metadata_cpu,
                key.cu_seqlens_values,
                key.feature_lens_values,
                key.aftercnn_lens_values,
                async_tensor_h2d,
            )

        with self._transaction_lock:
            output = self._backend.replay_and_clone(
                replay_entry,
                padded_feature,
            )
        if not self._logged_replay:
            logger.info(
                "ASR audio frontend CUDA graph replay active for shape=%s rows=%d",
                key.padded_shape,
                key.packed_rows,
            )
            self._logged_replay = True
        return output

    def _shared_capture_resources_locked(
        self,
        device: torch.device,
    ) -> tuple[Any, Any]:
        """Create one stream/private pool while the transaction lock is held."""
        if self._execution_stream is None and self._graph_pool is None:
            execution_stream, graph_pool = (
                self._backend.create_shared_capture_resources(device)
            )
            self._execution_stream = execution_stream
            self._graph_pool = graph_pool
        elif self._execution_stream is None or self._graph_pool is None:
            raise RuntimeError("Audio frontend graph capture state is incomplete")
        return self._execution_stream, self._graph_pool

    def _admit_existing_entry_locked(
        self,
        encoder: Any,
        padded_feature: torch.Tensor,
        chunk_lengths_cpu: torch.Tensor,
        pack_metadata_cpu: torch.Tensor,
        key: FrontendGraphKey,
        entry: _FrontendGraphEntry,
        async_tensor_h2d: Any,
    ) -> torch.Tensor | None:
        """Gate one alias while ``self._global_lock`` is held."""
        with self._transaction_lock:
            return self._admit_existing_entry_transaction_locked(
                encoder,
                padded_feature,
                chunk_lengths_cpu,
                pack_metadata_cpu,
                key,
                entry,
                async_tensor_h2d,
            )

    def _admit_existing_entry_transaction_locked(
        self,
        encoder: Any,
        padded_feature: torch.Tensor,
        chunk_lengths_cpu: torch.Tensor,
        pack_metadata_cpu: torch.Tensor,
        key: FrontendGraphKey,
        entry: _FrontendGraphEntry,
        async_tensor_h2d: Any,
    ) -> torch.Tensor | None:
        """Bitwise-gate a full key while both cache locks are held."""
        reference_output: torch.Tensor | None = None
        try:
            reference_output = self._eager(
                encoder,
                padded_feature,
                chunk_lengths_cpu,
                pack_metadata_cpu,
                key.cu_seqlens_values,
                key.feature_lens_values,
                key.aftercnn_lens_values,
                async_tensor_h2d,
            )
            candidate_output = self._backend.replay_and_clone(
                entry,
                padded_feature,
            )
            if not self._backend.equal(candidate_output, reference_output):
                raise RuntimeError(
                    "Shared audio frontend graph is not bitwise exact"
                )
            self._admitted_keys.add(key)
            logger.info(
                "Admitted exact audio frontend metadata onto existing CUDA "
                "graph shape=%s rows=%d feature_lens=%s observation=%d/%d "
                "signatures=%d/%d admitted_keys=%d",
                key.padded_shape,
                key.packed_rows,
                key.feature_lens_values,
                _PROBATION_OBSERVATIONS,
                _PROBATION_OBSERVATIONS,
                len(self._entries),
                self._max_entries,
                len(self._admitted_keys),
            )
            return reference_output
        except Exception as error:
            self._rejected_keys.add(key)
            logger.warning(
                "Audio frontend CUDA graph alias admission failed closed for "
                "shape=%s rows=%d feature_lens=%s: %s",
                key.padded_shape,
                key.packed_rows,
                key.feature_lens_values,
                error,
            )
            return reference_output

    def _capture_entry_locked(
        self,
        encoder: Any,
        padded_feature: torch.Tensor,
        chunk_lengths_cpu: torch.Tensor,
        pack_metadata_cpu: torch.Tensor,
        key: FrontendGraphKey,
        signature: FrontendGraphSignature,
        async_tensor_h2d: Any,
    ) -> tuple[_FrontendGraphEntry | None, torch.Tensor | None]:
        """Capture one signature while ``self._global_lock`` is held."""
        with self._transaction_lock:
            return self._capture_entry_transaction_locked(
                encoder,
                padded_feature,
                chunk_lengths_cpu,
                pack_metadata_cpu,
                key,
                signature,
                async_tensor_h2d,
            )

    def _capture_entry_transaction_locked(
        self,
        encoder: Any,
        padded_feature: torch.Tensor,
        chunk_lengths_cpu: torch.Tensor,
        pack_metadata_cpu: torch.Tensor,
        key: FrontendGraphKey,
        signature: FrontendGraphSignature,
        async_tensor_h2d: Any,
    ) -> tuple[_FrontendGraphEntry | None, torch.Tensor | None]:
        """Capture one signature while both cache locks are held."""
        reference_output: torch.Tensor | None = None
        try:
            static_padded_feature = torch.empty_strided(
                signature.padded_shape,
                signature.padded_stride,
                device=padded_feature.device,
                dtype=padded_feature.dtype,
            )
            static_pack_metadata = self._backend.make_device_metadata(
                pack_metadata_cpu,
                device=padded_feature.device,
                async_tensor_h2d=async_tensor_h2d,
            )
            static_padded_feature.copy_(padded_feature, non_blocking=True)

            def static_frontend() -> torch.Tensor:
                return run_audio_frontend_with_device_metadata(
                    encoder,
                    static_padded_feature,
                    static_pack_metadata,
                    total_rows=signature.packed_rows,
                )

            execution_stream, graph_pool = self._shared_capture_resources_locked(
                padded_feature.device
            )
            graph, graph_output, captured_stream, done_event = self._backend.capture(
                static_frontend,
                device=padded_feature.device,
                warmup_iterations=self._warmup_iterations,
                execution_stream=execution_stream,
                graph_pool=graph_pool,
            )
            if captured_stream is not execution_stream:
                raise RuntimeError("Audio frontend graph used a non-shared stream")
            if not _output_is_supported(graph_output, signature):
                raise RuntimeError("Captured audio frontend did not return [M, 1024]")

            reference_output = self._eager(
                encoder,
                padded_feature,
                chunk_lengths_cpu,
                pack_metadata_cpu,
                key.cu_seqlens_values,
                key.feature_lens_values,
                key.aftercnn_lens_values,
                async_tensor_h2d,
            )
            if not _output_is_supported(reference_output, key):
                raise RuntimeError("Eager audio frontend did not return [M, 1024]")
            self._backend.wait_stream(captured_stream, padded_feature.device)
            with self._backend.replay_context(captured_stream):
                static_padded_feature.copy_(padded_feature, non_blocking=True)
                self._backend.replay(graph)
                self._backend.record_done(done_event, captured_stream)
            self._backend.wait_done(done_event, padded_feature.device)
            if not self._backend.equal(graph_output, reference_output):
                raise RuntimeError("Captured audio frontend is not bitwise exact")

            entry = _FrontendGraphEntry(
                signature=signature,
                static_padded_feature=static_padded_feature,
                static_pack_metadata=static_pack_metadata,
                graph=graph,
                output=graph_output,
                execution_stream=captured_stream,
                done_event=done_event,
            )
            self._entries[signature] = entry
            self._admitted_keys.add(key)
            logger.info(
                "Captured bitwise-exact audio frontend CUDA graph shape=%s "
                "rows=%d feature_lens=%s observation=%d/%d signatures=%d/%d "
                "admitted_keys=%d",
                key.padded_shape,
                key.packed_rows,
                key.feature_lens_values,
                _PROBATION_OBSERVATIONS,
                _PROBATION_OBSERVATIONS,
                len(self._entries),
                self._max_entries,
                len(self._admitted_keys),
            )
            return entry, reference_output
        except Exception as error:
            self._rejected_keys.add(key)
            logger.warning(
                "Audio frontend CUDA graph capture failed closed for shape=%s "
                "rows=%d: %s",
                key.padded_shape,
                key.packed_rows,
                error,
            )
            return None, reference_output


class _NoopFrontendBackend:
    """CPU fake backend used by tests and the standalone helper's dry paths."""

    def supports(self, padded_feature: torch.Tensor) -> bool:
        return True

    def is_current_stream_capturing(self) -> bool:
        return False

    def make_device_metadata(
        self,
        metadata_cpu: torch.Tensor,
        *,
        device: torch.device,
        async_tensor_h2d: Any,
    ) -> torch.Tensor:
        del async_tensor_h2d
        return metadata_cpu.to(device=device)

    def create_shared_capture_resources(
        self,
        device: torch.device,
    ) -> tuple[Any, Any]:
        del device
        return object(), object()

    def capture(
        self,
        function: Callable[[], torch.Tensor],
        *,
        device: torch.device,
        warmup_iterations: int,
        execution_stream: Any,
        graph_pool: Any,
    ) -> tuple[Any, torch.Tensor, Any, Any]:
        del device, graph_pool
        for _ in range(warmup_iterations):
            function()
        output = function()
        return (
            {"function": function, "output": output},
            output,
            execution_stream,
            object(),
        )

    def replay_context(self, replay_stream: Any) -> Any:
        del replay_stream
        return nullcontext()

    def wait_stream(self, replay_stream: Any, device: torch.device) -> None:
        del replay_stream, device

    def record_done(self, done_event: Any, replay_stream: Any) -> None:
        del done_event, replay_stream

    def wait_done(self, done_event: Any, device: torch.device) -> None:
        del done_event, device

    def replay(self, graph: Any) -> None:
        graph["output"].copy_(graph["function"]())

    def replay_and_clone(
        self,
        entry: _FrontendGraphEntry,
        padded_feature: torch.Tensor,
    ) -> torch.Tensor:
        entry.static_padded_feature.copy_(padded_feature)
        self.replay(entry.graph)
        return entry.output.clone()

    def equal(self, left: torch.Tensor, right: torch.Tensor) -> bool:
        return torch.equal(left, right)


def run_audio_frontend_cudagraph(
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
    """Run an exact cached graph or the identical eager frontend."""
    cache = getattr(encoder, _CACHE_ATTR, None)
    if cache is None:
        with _CACHE_CREATION_LOCK:
            cache = getattr(encoder, _CACHE_ATTR, None)
            if cache is None:
                cache = ExactShapeAudioFrontendGraphCache()
                setattr(encoder, _CACHE_ATTR, cache)
    if not isinstance(cache, ExactShapeAudioFrontendGraphCache):
        return run_audio_frontend_eager(
            encoder,
            padded_feature,
            chunk_lengths_cpu,
            pack_metadata_cpu,
            cu_seqlens_values,
            feature_lens_values,
            aftercnn_lens_values,
            async_tensor_h2d=async_tensor_h2d,
        )
    return cache.run(
        encoder,
        padded_feature,
        chunk_lengths_cpu,
        pack_metadata_cpu,
        cu_seqlens_values,
        feature_lens_values,
        aftercnn_lens_values,
        async_tensor_h2d=async_tensor_h2d,
    )


def install_audio_frontend_cudagraph_patch() -> bool:
    """Install the frontend runner through the accepted CPU-metadata forward."""
    if not audio_frontend_cudagraph_enabled():
        return False
    if (
        os.environ.get(METADATA_ENV_NAME, "0") != "1"
        or os.environ.get(MAX_SEQLEN_ENV_NAME, "0") != "1"
    ):
        logger.warning(
            "%s requires %s=1 and %s=1; leaving the model unchanged",
            ENV_NAME,
            METADATA_ENV_NAME,
            MAX_SEQLEN_ENV_NAME,
        )
        return False

    from vllm.model_executor.models import qwen3_asr as asr_module
    from vllm.model_executor.models.qwen3_asr import Qwen3OmniMoeAudioEncoder

    current_forward = Qwen3OmniMoeAudioEncoder.forward
    current_field_config = asr_module._qwen3asr_field_config
    frontend_patched = bool(getattr(current_forward, _PATCH_MARKER, False))
    metadata_forward_patched = bool(
        getattr(current_forward, METADATA_PATCH_MARKER, False)
    )
    metadata_field_patched = bool(
        getattr(current_field_config, METADATA_PATCH_MARKER, False)
    )
    if frontend_patched:
        if metadata_forward_patched and metadata_field_patched:
            return True
        raise RuntimeError("Audio frontend CUDA graph patch is partially installed")
    if metadata_forward_patched or metadata_field_patched:
        logger.warning(
            "Audio CPU metadata patch was already installed without the frontend "
            "runner; refusing a partial replacement"
        )
        return False

    if not install_audio_cpu_metadata_pack_patch(
        frontend_runner=run_audio_frontend_cudagraph,
    ):
        return False
    setattr(Qwen3OmniMoeAudioEncoder.forward, _PATCH_MARKER, True)
    return True
