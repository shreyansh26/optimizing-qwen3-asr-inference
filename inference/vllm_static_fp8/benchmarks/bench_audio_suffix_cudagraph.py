#!/usr/bin/env python3
"""Gate the natural-audio row-bucket graph for the audio suffix on CUDA."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import statistics
import sys
import threading
import time
from typing import Callable

import torch

PATCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PATCH_DIR))

from audio_cpu_metadata_pack_patch import run_audio_suffix_eager  # noqa: E402
from audio_suffix_cudagraph_patch import (
    _MAX_CACHE_ENTRIES,
    _NATURAL_FULL_CHUNK_ROWS,
    _PROBATION_OBSERVATIONS,
    _SUPPORTED_ROWS,
    _TAIL_ROWS,
    _bucket_rows,
    _canonical_cu_seqlens_values,
    _make_bucket_key,
    _make_suffix_graph_key,
    ExactShapeAudioSuffixGraphCache,
)  # noqa: E402


_DEFAULT_CASES = tuple(
    (104, 104, rows - 208) for rows in sorted(_TAIL_ROWS)
) + tuple(
    (104, 104, 104, rows - 312)
    for rows in sorted(_NATURAL_FULL_CHUNK_ROWS)
)


def _parse_segments(raw_value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value) for value in raw_value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "segments must be comma-separated positive integers"
        ) from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(
            "segments must be comma-separated positive integers"
        )
    return values


def _cumulative_values(segments: tuple[int, ...]) -> tuple[int, ...]:
    cumulative = [0]
    for length in segments:
        cumulative.append(cumulative[-1] + length)
    return tuple(cumulative)


def _cu_metadata(
    segments: tuple[int, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    values = _cumulative_values(segments)
    cu_seqlens = torch.tensor(values, dtype=torch.int32, device=device)
    max_seqlen = torch.tensor(
        104,
        dtype=torch.int32,
        device="cpu",
    )
    return cu_seqlens, max_seqlen, values


def _initialize_single_gpu_distributed(port: int) -> None:
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        local_rank=0,
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )


def _new_encoder(device: torch.device) -> torch.nn.Module:
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoderConfig,
    )
    from vllm.model_executor.models.qwen3_asr import Qwen3OmniMoeAudioEncoder
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    config = Qwen3OmniMoeAudioEncoderConfig(
        num_mel_bins=128,
        encoder_layers=24,
        encoder_attention_heads=16,
        encoder_ffn_dim=4096,
        d_model=1024,
        output_dim=2048,
        activation_function="gelu",
        n_window=50,
        n_window_infer=800,
        conv_chunksize=500,
        downsample_hidden_size=480,
    )
    with set_default_torch_dtype(torch.bfloat16), torch.device(device):
        encoder = Qwen3OmniMoeAudioEncoder(
            config,
            prefix="audio_tower",
        )
    encoder.eval()
    if len(encoder.layers) != 24:
        raise RuntimeError("Expected the Qwen3-ASR 24-layer audio encoder")
    if encoder.attn_backend != AttentionBackendEnum.FLASH_ATTN:
        raise RuntimeError(
            f"Expected FLASH_ATTN audio backend, got {encoder.attn_backend}"
        )

    generator = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for name, parameter in encoder.named_parameters():
            if parameter.ndim == 1:
                if name.endswith("weight"):
                    parameter.fill_(1)
                else:
                    parameter.zero_()
            else:
                parameter.normal_(mean=0.0, std=0.01, generator=generator)
    return encoder


def _median_cuda_us(
    function: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeats: int,
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1_000)
        del output
    return statistics.median(samples)


def _median_host_call_us(
    function: Callable[[], torch.Tensor],
    *,
    repeats: int,
) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        output = function()
        samples.append((time.perf_counter_ns() - start) / 1_000)
        del output
    torch.cuda.synchronize()
    return statistics.median(samples)


def _cuda_kernel_names(function: Callable[[], object]) -> list[str]:
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        function()
        torch.cuda.synchronize()
    return [
        event.name
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]


def _select_shared_bucket_cases(
    cases: list[tuple[int, ...]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    by_bucket: dict[int, list[tuple[int, ...]]] = {}
    for segments in cases:
        bucket_rows = _bucket_rows(sum(segments))
        if bucket_rows is None:
            raise AssertionError("validated helper case has no row bucket")
        by_bucket.setdefault(bucket_rows, []).append(segments)
    shared = max(by_bucket.values(), key=len)
    if len(shared) < 2:
        raise ValueError(
            "The cross-key gates require two distinct cases in one row bucket"
        )
    return shared[0], shared[-1]


def _alternating_cross_key_gate(
    encoder: torch.nn.Module,
    cache: ExactShapeAudioSuffixGraphCache,
    *,
    segments_by_key: tuple[tuple[int, ...], tuple[int, ...]],
    device: torch.device,
    generator: torch.Generator,
) -> None:
    """Alternate retained outputs across two exact keys sharing one graph."""
    requests = []
    for request_index in range(4):
        segments = segments_by_key[request_index % 2]
        cu_seqlens, max_seqlen, cu_values = _cu_metadata(segments, device)
        hidden_states = torch.randn(
            (sum(segments), 1024),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        requests.append(
            (hidden_states, cu_seqlens, max_seqlen, cu_values)
        )

    with torch.inference_mode():
        expected = [
            run_audio_suffix_eager(
                encoder,
                hidden_states,
                cu_seqlens,
                max_seqlen,
                cu_seqlens_values=cu_values,
            )
            for hidden_states, cu_seqlens, max_seqlen, cu_values in requests
        ]
    torch.cuda.synchronize(device)
    with torch.inference_mode():
        outputs = [
            cache.run(
                encoder,
                hidden_states,
                cu_seqlens,
                max_seqlen,
                cu_seqlens_values=cu_values,
            )
            for hidden_states, cu_seqlens, max_seqlen, cu_values in requests
        ]
    torch.cuda.synchronize(device)

    for request_index, (output, reference) in enumerate(
        zip(outputs, expected, strict=True)
    ):
        if not torch.equal(output, reference):
            raise AssertionError(
                "alternating cross-key replay is not bitwise exact for "
                f"request={request_index}"
            )
    if len({output.data_ptr() for output in outputs}) != len(outputs):
        raise AssertionError("alternating replay outputs do not own storage")
    print(
        "alternating_cross_key_gate=PASS keys=2 requests=4 rows="
        f"{sum(segments_by_key[0])},{sum(segments_by_key[1])}"
    )


def _concurrent_cross_key_gate(
    encoder: torch.nn.Module,
    cache: ExactShapeAudioSuffixGraphCache,
    *,
    segments_by_thread: tuple[tuple[int, ...], tuple[int, ...]],
    device: torch.device,
    generator: torch.Generator,
    iterations: int,
) -> None:
    """Race two admitted keys sharing one bucket from two CUDA streams."""
    metadata_by_thread = [
        _cu_metadata(segments, device) for segments in segments_by_thread
    ]
    hidden_by_thread = [
        [
            torch.randn(
                (sum(segments), 1024),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            for _ in range(iterations)
        ]
        for segments in segments_by_thread
    ]
    with torch.inference_mode():
        expected_by_thread = [
            [
                run_audio_suffix_eager(
                    encoder,
                    hidden_states,
                    metadata_by_thread[thread_index][0],
                    metadata_by_thread[thread_index][1],
                    cu_seqlens_values=metadata_by_thread[thread_index][2],
                )
                for hidden_states in thread_inputs
            ]
            for thread_index, thread_inputs in enumerate(hidden_by_thread)
        ]
    # Finish input production and eager references before the worker streams
    # start. The race below therefore isolates cache replay transaction safety.
    torch.cuda.synchronize(device)

    streams = [torch.cuda.Stream(device=device) for _ in range(2)]
    if streams[0].cuda_stream == streams[1].cuda_stream:
        raise AssertionError("concurrency gate requires two distinct CUDA streams")
    barrier = threading.Barrier(2)

    def replay_many(
        thread_index: int,
        stream: torch.cuda.Stream,
    ) -> list[torch.Tensor]:
        torch.cuda.set_device(device)
        outputs: list[torch.Tensor] = []
        cu_seqlens, max_seqlen, cu_values = metadata_by_thread[thread_index]
        try:
            with torch.inference_mode(), torch.cuda.stream(stream):
                for hidden_states in hidden_by_thread[thread_index]:
                    barrier.wait()
                    outputs.append(
                        cache.run(
                            encoder,
                            hidden_states,
                            cu_seqlens,
                            max_seqlen,
                            cu_seqlens_values=cu_values,
                        )
                    )
            stream.synchronize()
            return outputs
        except BaseException:
            barrier.abort()
            raise

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(replay_many, thread_index, stream)
            for thread_index, stream in enumerate(streams)
        ]
        outputs_by_thread = [future.result() for future in futures]

    # Compare only after all iterations so this also proves later replays did
    # not overwrite an earlier caller's independently owned clone.
    for thread_index, (outputs, expected_outputs) in enumerate(
        zip(outputs_by_thread, expected_by_thread, strict=True)
    ):
        for iteration, (output, expected) in enumerate(
            zip(outputs, expected_outputs, strict=True)
        ):
            if not torch.equal(output, expected):
                raise AssertionError(
                    "concurrent suffix replay is not bitwise exact for "
                    f"thread={thread_index}, iteration={iteration}"
                )

    print(
        "concurrent_cross_key_gate=PASS keys=2 threads=2 streams=2 "
        f"iterations={iterations} rows="
        f"{sum(segments_by_thread[0])},{sum(segments_by_thread[1])}"
    )


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segments",
        action="append",
        type=_parse_segments,
        help=(
            "Comma-separated exact sequence lengths. Repeat for every observed "
            "key. Defaults cover all 14 natural keys in one row bucket."
        ),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--concurrency-iterations", type=int, default=5)
    parser.add_argument("--master-port", type=int, default=29617)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if (
        args.warmup <= 0
        or args.repeats <= 0
        or args.concurrency_iterations <= 0
    ):
        raise ValueError(
            "--warmup, --repeats, and --concurrency-iterations must be positive"
        )
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        raise RuntimeError("This production-shape gate requires an SM90 GPU")

    cases = args.segments or list(_DEFAULT_CASES)
    if any(max(segments) > 104 for segments in cases):
        raise ValueError(
            "Each segment must be <=104 to match the accepted CPU max-seqlen "
            "contract"
        )
    case_values = [_cumulative_values(segments) for segments in cases]
    if len(cases) > len(_SUPPORTED_ROWS) or any(
        values != _canonical_cu_seqlens_values(values[-1])
        for values in case_values
    ):
        raise ValueError(
            "Expected at most 14 canonical natural-chunk exact keys with a "
            f"supported total row count in {sorted(_SUPPORTED_ROWS)}"
        )
    if len(set(case_values)) != len(case_values):
        raise ValueError("Every exact canonical case must be distinct")
    cross_key_cases = _select_shared_bucket_cases(cases)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    _initialize_single_gpu_distributed(args.master_port)
    encoder = _new_encoder(device)
    cache = ExactShapeAudioSuffixGraphCache(
        max_entries=_MAX_CACHE_ENTRIES,
        warmup_iterations=args.warmup,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)

    print(
        "case,rows,sequences,bitwise_exact,bucket_kernel_order_exact,"
        "bucket_eager_kernels,replay_kernels,eager_cuda_us,"
        "replay_with_input_copy_cuda_us,cuda_speedup_pct,eager_host_call_us,"
        "replay_with_input_copy_host_call_us,host_speedup_pct"
    )
    with torch.inference_mode():
        for case_index, segments in enumerate(cases):
            cu_seqlens, max_seqlen, cu_values = _cu_metadata(segments, device)
            rows = sum(segments)
            initial_hidden = torch.randn(
                (rows, 1024),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            request_key = _make_suffix_graph_key(
                initial_hidden,
                cu_seqlens,
                max_seqlen,
                cu_values,
            )
            if request_key is None:
                raise AssertionError(
                    f"validated helper case has no key: {segments}"
                )
            bucket_key = _make_bucket_key(request_key)

            entries_before = cache.entry_count
            admissions_before = cache.admitted_count
            bucket_existed_before = bucket_key in cache._entries
            for observation in range(1, _PROBATION_OBSERVATIONS + 1):
                cache.run(
                    encoder,
                    initial_hidden,
                    cu_seqlens,
                    max_seqlen,
                    cu_seqlens_values=cu_values,
                )
                expected_entries = entries_before + (
                    observation == _PROBATION_OBSERVATIONS
                    and not bucket_existed_before
                )
                if cache.entry_count != expected_entries:
                    raise AssertionError(
                        "bucket capture count changed outside observation 8; "
                        f"case={segments}, "
                        f"observation={observation}"
                    )
                expected_admissions = admissions_before + (
                    observation == _PROBATION_OBSERVATIONS
                )
                if cache.admitted_count != expected_admissions:
                    raise AssertionError(
                        "exact-key probation must admit only on observation 8; "
                        f"case={segments}, observation={observation}"
                    )
            replay_hidden = torch.randn(
                (rows, 1024),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            eager_output = run_audio_suffix_eager(
                encoder,
                replay_hidden,
                cu_seqlens,
                max_seqlen,
                cu_seqlens_values=cu_values,
            )
            replay_output = cache.run(
                encoder,
                replay_hidden,
                cu_seqlens,
                max_seqlen,
                cu_seqlens_values=cu_values,
            )
            torch.cuda.synchronize()
            bitwise_exact = torch.equal(eager_output, replay_output)
            if not bitwise_exact:
                raise AssertionError(
                    f"suffix replay is not bitwise exact for {segments}"
                )

            entry = cache._entries[bucket_key]
            entry.static_hidden_states[:rows].copy_(replay_hidden)
            entry.static_cu_seqlens[:-1].copy_(cu_seqlens)
            torch.cuda.synchronize()
            bucket_eager_kernels = _cuda_kernel_names(
                lambda: run_audio_suffix_eager(
                    encoder,
                    entry.static_hidden_states,
                    entry.static_cu_seqlens,
                    entry.static_max_seqlen,
                    cu_seqlens_values=(*cu_values, bucket_key.bucket_rows),
                )
            )
            replay_kernels = _cuda_kernel_names(
                lambda: cache._backend.replay(entry.graph)
            )
            bucket_kernel_order_exact = bucket_eager_kernels == replay_kernels
            if not bucket_kernel_order_exact:
                raise AssertionError(
                    "captured bucket kernel names/order differ from bucket "
                    "eager for "
                    f"{segments}"
                )

            eager_call = lambda: run_audio_suffix_eager(  # noqa: E731
                encoder,
                replay_hidden,
                cu_seqlens,
                max_seqlen,
                cu_seqlens_values=cu_values,
            )
            replay_call = lambda: cache.run(  # noqa: E731
                encoder,
                replay_hidden,
                cu_seqlens,
                max_seqlen,
                cu_seqlens_values=cu_values,
            )
            eager_us = _median_cuda_us(
                eager_call,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            replay_us = _median_cuda_us(
                replay_call,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            eager_host_us = _median_host_call_us(
                eager_call,
                repeats=args.repeats,
            )
            replay_host_us = _median_host_call_us(
                replay_call,
                repeats=args.repeats,
            )
            cuda_speedup_pct = (eager_us / replay_us - 1) * 100
            host_speedup_pct = (eager_host_us / replay_host_us - 1) * 100
            print(
                f"{case_index},{rows},{len(segments)},{bitwise_exact},"
                f"{bucket_kernel_order_exact},{len(bucket_eager_kernels)},"
                f"{len(replay_kernels)},{eager_us:.3f},{replay_us:.3f},"
                f"{cuda_speedup_pct:.3f},{eager_host_us:.3f},"
                f"{replay_host_us:.3f},{host_speedup_pct:.3f}"
            )

    expected_bucket_count = len(
        {_bucket_rows(sum(segments)) for segments in cases}
    )
    if cache.entry_count != expected_bucket_count:
        raise AssertionError(
            f"expected {expected_bucket_count} bucket graphs, "
            f"got {cache.entry_count}"
        )
    if cache.admitted_count != len(cases):
        raise AssertionError(
            f"expected {len(cases)} exact admissions, "
            f"got {cache.admitted_count}"
        )
    _alternating_cross_key_gate(
        encoder,
        cache,
        segments_by_key=cross_key_cases,
        device=device,
        generator=generator,
    )
    _concurrent_cross_key_gate(
        encoder,
        cache,
        segments_by_thread=cross_key_cases,
        device=device,
        generator=generator,
        iterations=args.concurrency_iterations,
    )
    print("gate=PASS_BUCKETED_AUDIO_SUFFIX_CUDAGRAPH")


def main() -> None:
    # Direct module construction needs the same config context that vLLM's
    # production model loader installs around initialization and execution.
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    with set_current_vllm_config(VllmConfig()):
        try:
            _main()
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


if __name__ == "__main__":
    main()
