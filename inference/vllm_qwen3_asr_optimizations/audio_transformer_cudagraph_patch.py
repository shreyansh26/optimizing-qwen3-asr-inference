"""Exact-admitted row-bucket CUDA graphs for the Qwen3-ASR audio transformer."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import threading
import time
from typing import Any, Callable

import torch

from audio_cpu_maxseqlen_patch import STATIC_MAX_SEQLEN
from audio_cpu_metadata_pack_patch import (
    ENV_NAME as METADATA_ENV_NAME,
    MAX_SEQLEN_ENV_NAME,
    _PATCH_MARKER as METADATA_PATCH_MARKER,
    install_audio_cpu_metadata_pack_patch,
    run_audio_transformer_eager,
)
from vllm.logger import init_logger


logger = init_logger("vllm.qwen3_asr_audio_transformer_cudagraph")

ENV_NAME = "ASR_AUDIO_TRANSFORMER_CUDAGRAPH"
_PATCH_MARKER = "_asr_audio_transformer_cudagraph_patch"
_CACHE_ATTR = "_asr_audio_transformer_cudagraph_cache"
_CACHE_CREATION_LOCK = threading.Lock()

# Exact transformer contract after the frontend pack:
#
#   hidden_states: [rows, 1024]
#   cu_seqlens:    [0, 104, 208, 312, rows] for admitted 377--390 rows
#   max_seqlen:    scalar int32 CPU tensor containing 104
#   output:         [rows, 2048]
#
# Frontend geometry supplies 13 rows per full 100-frame chunk. Natural
# 29--30-chunk inputs therefore contain 29*13=377 through 30*13=390 rows.
_EXPECTED_LAYER_COUNT = 24
_INPUT_WIDTH = 1024
_OUTPUT_WIDTH = 2 * _INPUT_WIDTH
_EXPECTED_MAX_SEQLEN = STATIC_MAX_SEQLEN
_RAW_CHUNK_FRAMES = 100
_CONV_DOWNSAMPLE_FACTOR = 1 << 3  # three stride-2 convolutions
_ROWS_PER_FULL_CHUNK = (
    _RAW_CHUNK_FRAMES + _CONV_DOWNSAMPLE_FACTOR - 1
) // _CONV_DOWNSAMPLE_FACTOR
_MIN_NATURAL_CHUNKS = 29
_MAX_NATURAL_CHUNKS = 30
_TAIL_ROWS = frozenset()
_NATURAL_FULL_CHUNK_ROWS = frozenset(
    range(
        _MIN_NATURAL_CHUNKS * _ROWS_PER_FULL_CHUNK,
        _MAX_NATURAL_CHUNKS * _ROWS_PER_FULL_CHUNK + 1,
    )
)
_SUPPORTED_ROWS = _NATURAL_FULL_CHUNK_ROWS
_NATURAL_BUCKET_ROWS = _MAX_NATURAL_CHUNKS * _ROWS_PER_FULL_CHUNK
_MAX_CACHE_ENTRIES = 1
_WARMUP_ITERATIONS = 3
_PROBATION_OBSERVATIONS = 8


@dataclass(frozen=True)
class TransformerGraphKey:
    """Exact runtime state that must pass its own probation and admission."""

    rows: int
    cu_seqlens_numel: int
    dtype: torch.dtype
    device_type: str
    device_index: int | None
    max_seqlen_value: int
    cu_seqlens_values: tuple[int, ...]


@dataclass(frozen=True)
class TransformerBucketKey:
    """Graph-static state shared by one canonical row-count family."""

    bucket_rows: int
    graph_cu_seqlens_numel: int
    dtype: torch.dtype
    device_type: str
    device_index: int | None
    max_seqlen_value: int


@dataclass
class _TransformerGraphEntry:
    key: TransformerBucketKey
    static_hidden_states: torch.Tensor
    static_cu_seqlens: torch.Tensor
    static_max_seqlen: torch.Tensor
    graph: Any
    output: torch.Tensor
    execution_stream: Any
    replay_count: int = 0
    replay_lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
    )


def audio_transformer_cudagraph_enabled() -> bool:
    """Return the strict boolean environment gate for this experiment."""
    raw_value = os.environ.get(ENV_NAME, "0")
    if raw_value not in {"0", "1"}:
        raise ValueError(f"{ENV_NAME} must be 0 or 1, got {raw_value!r}")
    return raw_value == "1"


def _canonical_cu_seqlens_values(rows: int) -> tuple[int, ...] | None:
    if rows in _NATURAL_FULL_CHUNK_ROWS:
        # range() yields the complete 104-row attention windows; appending rows
        # adds the final 65--78-row tail endpoint.
        return (*range(0, rows, _EXPECTED_MAX_SEQLEN), rows)
    return None


def _bucket_rows(rows: int) -> int | None:
    if rows in _NATURAL_FULL_CHUNK_ROWS:
        return _NATURAL_BUCKET_ROWS
    return None


def _make_bucket_key(key: TransformerGraphKey) -> TransformerBucketKey:
    bucket_rows = _bucket_rows(key.rows)
    if bucket_rows is None:
        raise ValueError("Unsupported exact transformer key cannot select a bucket")
    return TransformerBucketKey(
        bucket_rows=bucket_rows,
        # The graph adds its fixed bucket endpoint after the runtime
        # cu_seqlens, including an intentional duplicate when rows == 390.
        graph_cu_seqlens_numel=key.cu_seqlens_numel + 1,
        dtype=key.dtype,
        device_type=key.device_type,
        device_index=key.device_index,
        max_seqlen_value=key.max_seqlen_value,
    )


def _graph_cu_seqlens_values(key: TransformerGraphKey) -> tuple[int, ...]:
    bucket_rows = _bucket_rows(key.rows)
    if bucket_rows is None:
        raise ValueError("Unsupported exact transformer key cannot select a bucket")
    # Always append the endpoint so each family has one fixed metadata shape.
    # For M == bucket this is an intentional zero-length final sequence; exact
    # admission and the CUDA helper must validate that installed FA3 path.
    return (*key.cu_seqlens_values, bucket_rows)


def _make_transformer_graph_key(
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: torch.Tensor | None,
    cu_seqlens_values: tuple[int, ...],
) -> TransformerGraphKey | None:
    if (
        not isinstance(hidden_states, torch.Tensor)
        or hidden_states.ndim != 2
        or hidden_states.shape[0] <= 0
        or hidden_states.shape[1] != _INPUT_WIDTH
        or not hidden_states.is_contiguous()
        or not isinstance(cu_seqlens, torch.Tensor)
        or cu_seqlens.ndim != 1
        or cu_seqlens.dtype != torch.int32
        or not cu_seqlens.is_contiguous()
        or cu_seqlens.device != hidden_states.device
        or not isinstance(max_seqlen, torch.Tensor)
        or max_seqlen.ndim != 0
        or max_seqlen.dtype != torch.int32
        or max_seqlen.device.type != "cpu"
    ):
        return None

    values = tuple(int(value) for value in cu_seqlens_values)
    rows = hidden_states.shape[0]
    canonical_values = _canonical_cu_seqlens_values(rows)
    max_seqlen_value = int(max_seqlen.item())
    if (
        len(values) != cu_seqlens.numel()
        or canonical_values is None
        or values != canonical_values
        or max_seqlen_value != _EXPECTED_MAX_SEQLEN
    ):
        return None

    return TransformerGraphKey(
        rows=rows,
        cu_seqlens_numel=cu_seqlens.numel(),
        dtype=hidden_states.dtype,
        device_type=hidden_states.device.type,
        device_index=hidden_states.device.index,
        max_seqlen_value=max_seqlen_value,
        cu_seqlens_values=values,
    )


def _output_is_supported(
    output: Any,
    *,
    rows: int,
    dtype: torch.dtype,
    device_type: str,
    device_index: int | None,
) -> bool:
    return (
        isinstance(output, torch.Tensor)
        and output.shape == (rows, _OUTPUT_WIDTH)
        and output.dtype == dtype
        and output.device.type == device_type
        and output.device.index == device_index
    )


class _TorchCudaGraphBackend:
    """Small CUDA API seam so cache behavior has CUDA-hidden fake tests."""

    def supports(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
    ) -> bool:
        return (
            torch.cuda.is_available()
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and cu_seqlens.is_cuda
            and max_seqlen.device.type == "cpu"
            and max_seqlen.dtype == torch.int32
        )

    def is_current_stream_capturing(self) -> bool:
        return torch.cuda.is_current_stream_capturing()

    def capture(
        self,
        function: Callable[[], torch.Tensor],
        *,
        device: torch.device,
        warmup_iterations: int,
    ) -> tuple[Any, torch.Tensor, Any]:
        caller_stream = torch.cuda.current_stream(device)
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(caller_stream)
        with torch.cuda.stream(capture_stream):
            for _ in range(warmup_iterations):
                function()
        caller_stream.wait_stream(capture_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph,
            stream=capture_stream,
            capture_error_mode="thread_local",
        ):
            output = function()
        return graph, output, capture_stream

    def replay(self, graph: Any) -> None:
        graph.replay()

    def replay_and_clone(
        self,
        entry: _TransformerGraphEntry,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rows: int,
    ) -> torch.Tensor:
        caller_stream = torch.cuda.current_stream(hidden_states.device)
        execution_stream = entry.execution_stream

        # The dedicated entry stream consumes the caller's input only after
        # its producer work. All mutable graph buffers and the output clone are
        # then ordered FIFO on this one stream.
        execution_stream.wait_stream(caller_stream)
        with torch.cuda.stream(execution_stream):
            entry.static_hidden_states[:rows].copy_(
                hidden_states,
                non_blocking=True,
            )
            entry.static_cu_seqlens[:-1].copy_(
                cu_seqlens,
                non_blocking=True,
            )
            hidden_states.record_stream(execution_stream)
            cu_seqlens.record_stream(execution_stream)
            entry.graph.replay()
            output = entry.output[:rows].clone()

        # Insert an asynchronous dependency back to the caller. record_stream
        # keeps the clone's storage alive if the caller consumes it after this
        # Python frame returns. Neither operation synchronizes the host.
        caller_stream.wait_stream(execution_stream)
        output.record_stream(caller_stream)
        return output

    def equal(self, left: torch.Tensor, right: torch.Tensor) -> bool:
        return torch.equal(left, right)


class ExactShapeAudioTransformerGraphCache:
    """Share one natural-row graph after exact per-request admission."""

    def __init__(
        self,
        *,
        backend: Any | None = None,
        max_entries: int = _MAX_CACHE_ENTRIES,
        warmup_iterations: int = _WARMUP_ITERATIONS,
    ) -> None:
        if max_entries <= 0 or warmup_iterations <= 0:
            raise ValueError("CUDA graph cache limits must be positive")
        self._backend = backend or _TorchCudaGraphBackend()
        self._max_entries = max_entries
        self._warmup_iterations = warmup_iterations
        self._entries: dict[TransformerBucketKey, _TransformerGraphEntry] = {}
        self._admitted_keys: dict[TransformerGraphKey, TransformerBucketKey] = {}
        self._observation_counts: dict[TransformerGraphKey, int] = {}
        self._rejected_keys: set[TransformerGraphKey] = set()
        self._logged_capacity = False
        # This lock protects the cache state and intentionally remains held
        # through cold capture/admission, so concurrent observation-eight calls
        # cannot capture/insert the same bucket twice. Hot GPU transactions use
        # per-entry locks shared by all exact keys in that bucket.
        self._cache_lock = threading.Lock()

    @property
    def entry_count(self) -> int:
        with self._cache_lock:
            return len(self._entries)

    @property
    def rejected_count(self) -> int:
        with self._cache_lock:
            return len(self._rejected_keys)

    @property
    def admitted_count(self) -> int:
        with self._cache_lock:
            return len(self._admitted_keys)

    def _eager(
        self,
        encoder: Any,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        cu_seqlens_values: tuple[int, ...],
    ) -> torch.Tensor:
        return run_audio_transformer_eager(
            encoder,
            hidden_states,
            cu_seqlens,
            max_seqlen,
            cu_seqlens_values=cu_seqlens_values,
        )

    def _replay_entry(
        self,
        entry: _TransformerGraphEntry,
        key: TransformerGraphKey,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        # Holding the lock until the complete transaction has been enqueued is
        # sufficient: every transaction uses entry.execution_stream, so CUDA
        # FIFO ordering completes the prior output clone before the next input
        # copy can overwrite graph-owned buffers. No device sync is needed.
        with entry.replay_lock:
            output = self._backend.replay_and_clone(
                entry,
                hidden_states,
                cu_seqlens,
                key.rows,
            )
            entry.replay_count += 1
            replay_count = entry.replay_count
            # Power-of-two milestones expose a precise cumulative counter while
            # avoiding a log operation on every admitted hot replay.
            if (replay_count & (replay_count - 1)) == 0:
                logger.info(
                    "ASR bucketed audio transformer CUDA graph replay active "
                    "cu_seqlens=%s bucket_rows=%d observation=%d/%d "
                    "bucket_cumulative_replays=%d",
                    key.cu_seqlens_values,
                    entry.key.bucket_rows,
                    _PROBATION_OBSERVATIONS,
                    _PROBATION_OBSERVATIONS,
                    replay_count,
                )
        return output

    def _capture_bucket_entry_locked(
        self,
        encoder: Any,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        key: TransformerGraphKey,
        bucket_key: TransformerBucketKey,
    ) -> _TransformerGraphEntry | None:
        """Capture one family bucket while ``self._cache_lock`` is held."""
        capture_start_ns = time.perf_counter_ns()
        graph_values = _graph_cu_seqlens_values(key)
        try:
            static_hidden_states = hidden_states.new_zeros(
                (bucket_key.bucket_rows, _INPUT_WIDTH)
            )
            static_hidden_states[: key.rows].copy_(
                hidden_states,
                non_blocking=True,
            )
            static_cu_seqlens = torch.empty(
                (bucket_key.graph_cu_seqlens_numel,),
                dtype=cu_seqlens.dtype,
                device=cu_seqlens.device,
            )
            static_cu_seqlens[:-1].copy_(
                cu_seqlens,
                non_blocking=True,
            )
            static_cu_seqlens[-1].fill_(bucket_key.bucket_rows)
            static_max_seqlen = torch.empty_like(max_seqlen)
            static_max_seqlen.copy_(max_seqlen, non_blocking=True)

            def static_transformer() -> torch.Tensor:
                return self._eager(
                    encoder,
                    static_hidden_states,
                    static_cu_seqlens,
                    static_max_seqlen,
                    graph_values,
                )

            graph, graph_output, execution_stream = self._backend.capture(
                static_transformer,
                device=hidden_states.device,
                warmup_iterations=self._warmup_iterations,
            )
            if not _output_is_supported(
                graph_output,
                rows=bucket_key.bucket_rows,
                dtype=bucket_key.dtype,
                device_type=bucket_key.device_type,
                device_index=bucket_key.device_index,
            ):
                raise RuntimeError(
                    "Captured bucketed audio transformer did not return "
                    f"[bucket_rows, {_OUTPUT_WIDTH}]"
                )

            entry = _TransformerGraphEntry(
                key=bucket_key,
                static_hidden_states=static_hidden_states,
                static_cu_seqlens=static_cu_seqlens,
                static_max_seqlen=static_max_seqlen,
                graph=graph,
                output=graph_output,
                execution_stream=execution_stream,
            )
            self._entries[bucket_key] = entry
            capture_duration_ms = (
                time.perf_counter_ns() - capture_start_ns
            ) / 1_000_000
            logger.info(
                "Captured bucketed audio transformer CUDA graph graph_cu_seqlens=%s "
                "bucket_rows=%d observation=%d/%d bucket_occupancy=%d/%d "
                "capture_duration_ms=%.3f cumulative_replays=0",
                graph_values,
                bucket_key.bucket_rows,
                _PROBATION_OBSERVATIONS,
                _PROBATION_OBSERVATIONS,
                len(self._entries),
                self._max_entries,
                capture_duration_ms,
            )
            return entry
        except Exception as error:
            capture_duration_ms = (
                time.perf_counter_ns() - capture_start_ns
            ) / 1_000_000
            logger.warning(
                "Bucketed audio transformer CUDA graph capture failed closed "
                "cu_seqlens=%s bucket_rows=%d observation=%d/%d "
                "bucket_occupancy=%d/%d "
                "capture_duration_ms=%.3f cumulative_replays=0: %s",
                key.cu_seqlens_values,
                bucket_key.bucket_rows,
                _PROBATION_OBSERVATIONS,
                _PROBATION_OBSERVATIONS,
                len(self._entries),
                self._max_entries,
                capture_duration_ms,
                error,
            )
            return None

    def _admit_key_locked(
        self,
        encoder: Any,
        entry: _TransformerGraphEntry,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        key: TransformerGraphKey,
    ) -> torch.Tensor:
        """Bitwise-gate one exact runtime key against unpadded eager."""
        reference_output: torch.Tensor | None = None
        try:
            reference_output = self._eager(
                encoder,
                hidden_states,
                cu_seqlens,
                max_seqlen,
                key.cu_seqlens_values,
            )
            if not _output_is_supported(
                reference_output,
                rows=key.rows,
                dtype=key.dtype,
                device_type=key.device_type,
                device_index=key.device_index,
            ):
                raise RuntimeError("Eager audio transformer did not return [M, 2048]")
            candidate_output = self._replay_entry_without_log(
                entry,
                key,
                hidden_states,
                cu_seqlens,
            )
            if not _output_is_supported(
                candidate_output,
                rows=key.rows,
                dtype=key.dtype,
                device_type=key.device_type,
                device_index=key.device_index,
            ):
                raise RuntimeError(
                    "Bucketed audio transformer replay did not return [M, 2048]"
                )
            # This comparison synchronizes once for each exact M. Admitted hot
            # replays never compare or synchronize on the host.
            if not self._backend.equal(candidate_output, reference_output):
                raise RuntimeError(
                    "Bucketed audio transformer is not bitwise equal to unpadded eager"
                )

            self._admitted_keys[key] = entry.key
            logger.info(
                "Admitted bitwise-exact bucketed audio transformer "
                "cu_seqlens=%s graph_cu_seqlens=%s bucket_rows=%d "
                "observation=%d/%d admitted=%d/%d",
                key.cu_seqlens_values,
                _graph_cu_seqlens_values(key),
                entry.key.bucket_rows,
                _PROBATION_OBSERVATIONS,
                _PROBATION_OBSERVATIONS,
                len(self._admitted_keys),
                len(_SUPPORTED_ROWS),
            )
            # Observation eight returns the independently allocated eager
            # result. Graph-owned output is cloned only on subsequent hits.
            return reference_output
        except Exception as error:
            self._rejected_keys.add(key)
            logger.warning(
                "Bucketed audio transformer admission failed closed "
                "cu_seqlens=%s graph_cu_seqlens=%s bucket_rows=%d "
                "observation=%d/%d admitted=%d/%d: %s",
                key.cu_seqlens_values,
                _graph_cu_seqlens_values(key),
                entry.key.bucket_rows,
                _PROBATION_OBSERVATIONS,
                _PROBATION_OBSERVATIONS,
                len(self._admitted_keys),
                len(_SUPPORTED_ROWS),
                error,
            )
            if reference_output is not None:
                return reference_output
            return self._eager(
                encoder,
                hidden_states,
                cu_seqlens,
                max_seqlen,
                key.cu_seqlens_values,
            )

    def _replay_entry_without_log(
        self,
        entry: _TransformerGraphEntry,
        key: TransformerGraphKey,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        with entry.replay_lock:
            return self._backend.replay_and_clone(
                entry,
                hidden_states,
                cu_seqlens,
                key.rows,
            )

    def run(
        self,
        encoder: Any,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
        *,
        cu_seqlens_values: tuple[int, ...],
    ) -> torch.Tensor:
        key = _make_transformer_graph_key(
            hidden_states,
            cu_seqlens,
            max_seqlen,
            cu_seqlens_values,
        )
        if (
            key is None
            or max_seqlen is None
            or len(getattr(encoder, "layers", ())) != _EXPECTED_LAYER_COUNT
            or bool(getattr(encoder, "training", True))
            or torch.is_grad_enabled()
            or not self._backend.supports(
                hidden_states,
                cu_seqlens,
                max_seqlen,
            )
            or self._backend.is_current_stream_capturing()
        ):
            return self._eager(
                encoder,
                hidden_states,
                cu_seqlens,
                max_seqlen,
                tuple(cu_seqlens_values),
            )

        bucket_key = _make_bucket_key(key)
        with self._cache_lock:
            key_rejected = key in self._rejected_keys
            admitted_bucket = self._admitted_keys.get(key)
            entry = self._entries.get(bucket_key)
            key_admitted = admitted_bucket == bucket_key and entry is not None
            cache_full = entry is None and len(self._entries) >= self._max_entries
            if cache_full and not self._logged_capacity:
                logger.warning(
                    "Bucketed audio transformer CUDA graph cache is full; "
                    "cu_seqlens=%s bucket_rows=%d observation=%d/%d "
                    "bucket_occupancy=%d/%d; using eager transformer",
                    key.cu_seqlens_values,
                    bucket_key.bucket_rows,
                    self._observation_counts.get(key, 0),
                    _PROBATION_OBSERVATIONS,
                    len(self._entries),
                    self._max_entries,
                )
                self._logged_capacity = True
            if not key_rejected and not key_admitted and not cache_full:
                observation_count = self._observation_counts.get(key, 0) + 1
                self._observation_counts[key] = observation_count
                if observation_count == _PROBATION_OBSERVATIONS:
                    if entry is None:
                        entry = self._capture_bucket_entry_locked(
                            encoder,
                            hidden_states,
                            cu_seqlens,
                            max_seqlen,
                            key,
                            bucket_key,
                        )
                    if entry is None:
                        self._rejected_keys.add(key)
                        return self._eager(
                            encoder,
                            hidden_states,
                            cu_seqlens,
                            max_seqlen,
                            key.cu_seqlens_values,
                        )
                    return self._admit_key_locked(
                        encoder,
                        entry,
                        hidden_states,
                        cu_seqlens,
                        max_seqlen,
                        key,
                    )
                logger.info(
                    "Bucketed audio transformer CUDA graph probation uses eager "
                    "transformer cu_seqlens=%s bucket_rows=%d observation=%d/%d "
                    "bucket_occupancy=%d/%d admitted=%d/%d",
                    key.cu_seqlens_values,
                    bucket_key.bucket_rows,
                    observation_count,
                    _PROBATION_OBSERVATIONS,
                    len(self._entries),
                    self._max_entries,
                    len(self._admitted_keys),
                    len(_SUPPORTED_ROWS),
                )

        if key_rejected:
            return self._eager(
                encoder,
                hidden_states,
                cu_seqlens,
                max_seqlen,
                key.cu_seqlens_values,
            )

        if key_admitted and entry is not None:
            return self._replay_entry(
                entry,
                key,
                hidden_states,
                cu_seqlens,
            )

        return self._eager(
            encoder,
            hidden_states,
            cu_seqlens,
            max_seqlen,
            key.cu_seqlens_values,
        )


def run_audio_transformer_cudagraph(
    encoder: Any,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: torch.Tensor | None,
    *,
    cu_seqlens_values: tuple[int, ...],
) -> torch.Tensor:
    """Run an exact-admitted row-bucket graph or the unpadded eager transformer."""
    cache = getattr(encoder, _CACHE_ATTR, None)
    if cache is None:
        # The cache-map lock cannot help until one cache is attached. Serialize
        # that one-time attachment so concurrent first calls cannot capture into
        # different cache objects and discard one of them.
        with _CACHE_CREATION_LOCK:
            cache = getattr(encoder, _CACHE_ATTR, None)
            if cache is None:
                cache = ExactShapeAudioTransformerGraphCache()
                setattr(encoder, _CACHE_ATTR, cache)
    if not isinstance(cache, ExactShapeAudioTransformerGraphCache):
        return run_audio_transformer_eager(
            encoder,
            hidden_states,
            cu_seqlens,
            max_seqlen,
            cu_seqlens_values=cu_seqlens_values,
        )
    return cache.run(
        encoder,
        hidden_states,
        cu_seqlens,
        max_seqlen,
        cu_seqlens_values=cu_seqlens_values,
    )


def install_audio_transformer_cudagraph_patch() -> bool:
    """Install the transformer runner through the accepted CPU-metadata forward."""
    if not audio_transformer_cudagraph_enabled():
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
    transformer_patched = bool(getattr(current_forward, _PATCH_MARKER, False))
    metadata_forward_patched = bool(
        getattr(current_forward, METADATA_PATCH_MARKER, False)
    )
    metadata_field_patched = bool(
        getattr(current_field_config, METADATA_PATCH_MARKER, False)
    )
    if transformer_patched:
        if metadata_forward_patched and metadata_field_patched:
            return True
        raise RuntimeError("Audio transformer CUDA graph patch is partially installed")
    if metadata_forward_patched or metadata_field_patched:
        logger.warning(
            "Audio CPU metadata patch was already installed without the transformer "
            "runner; refusing a partial replacement"
        )
        return False

    if not install_audio_cpu_metadata_pack_patch(
        transformer_runner=run_audio_transformer_cudagraph
    ):
        return False
    setattr(Qwen3OmniMoeAudioEncoder.forward, _PATCH_MARKER, True)
    return True
