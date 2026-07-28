#!/usr/bin/env python3
"""CUDA gate for exact natural-hotset Qwen3-ASR prefix graphs."""

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
from torch import nn

PATCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PATCH_DIR))

from audio_cpu_metadata_pack_patch import run_audio_prefix_eager  # noqa: E402
from audio_prefix_cudagraph_patch import (
    _MAX_CACHE_ENTRIES,
    _NATURAL_PACKED_ROWS,
    _PROBATION_OBSERVATIONS,
    _canonical_single_audio_natural_metadata,
    _make_prefix_graph_key,
    _make_prefix_graph_signature,
    _natural_feature_lengths_for_rows,
    ExactShapeAudioPrefixGraphCache,
    run_audio_prefix_with_device_metadata,
)  # noqa: E402
from vllm.utils.torch_utils import async_tensor_h2d  # noqa: E402


class PrefixOnlyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.training = False
        self.conv_chunksize = 500
        self.conv2d1 = nn.Conv2d(
            1,
            480,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            dtype=torch.bfloat16,
        )
        self.conv2d2 = nn.Conv2d(
            480,
            480,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            dtype=torch.bfloat16,
        )
        self.conv2d3 = nn.Conv2d(
            480,
            480,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            dtype=torch.bfloat16,
        )
        self.conv_out = nn.Linear(480 * 16, 1024, dtype=torch.bfloat16)
        self.positional_embedding = _PositionalEmbedding()


class _PositionalEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(13, 1024, dtype=torch.bfloat16),
            requires_grad=False,
        )


def _metadata(feature_length: int) -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    values = _canonical_single_audio_natural_metadata(feature_length)
    if values is None:
        raise ValueError("feature length is outside the natural hotset")
    chunk_lengths, pack_metadata, cu, feature_lens, aftercnn_lens = values
    return (
        torch.tensor(chunk_lengths, dtype=torch.int64),
        torch.tensor(pack_metadata, dtype=torch.int32),
        cu,
        feature_lens,
        aftercnn_lens,
    )


def _padded(
    feature_length: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    chunk_lengths = _metadata(feature_length)[0].tolist()
    chunks = [
        torch.randn(
            (length, 128),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        for length in chunk_lengths
    ]
    padded = torch.nn.utils.rnn.pad_sequence(
        chunks,
        batch_first=True,
    ).transpose(1, 2).unsqueeze(1)
    if padded.stride() != (12800, 128, 1, 128):
        raise AssertionError(
            f"unexpected production padding stride {padded.stride()}"
        )
    return padded


def _median_cuda_us(
    function: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeats: int,
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
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
    samples = []
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


def _advance_exact_key_to_admission(
    cache: ExactShapeAudioPrefixGraphCache,
    encoder: PrefixOnlyEncoder,
    padded: torch.Tensor,
    metadata: tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ],
    *,
    expected_entry_delta: int,
    expected_admitted_delta: int,
) -> None:
    entries_before = cache.entry_count
    admitted_before = cache.admitted_key_count
    for observation in range(1, _PROBATION_OBSERVATIONS + 1):
        cache.run(
            encoder,
            padded,
            *metadata,
            async_tensor_h2d=async_tensor_h2d,
        )
        expected_entries = entries_before + (
            expected_entry_delta
            if observation == _PROBATION_OBSERVATIONS
            else 0
        )
        expected_admitted = admitted_before + (
            expected_admitted_delta
            if observation == _PROBATION_OBSERVATIONS
            else 0
        )
        if cache.entry_count != expected_entries:
            raise AssertionError(
                "prefix probation changed graph occupancy before observation 8"
            )
        if cache.admitted_key_count != expected_admitted:
            raise AssertionError(
                "prefix probation admitted a full key before observation 8"
            )


def _concurrent_same_key_gate(
    encoder: PrefixOnlyEncoder,
    cache: ExactShapeAudioPrefixGraphCache,
    *,
    rows: int,
    device: torch.device,
    generator: torch.Generator,
    iterations: int,
) -> None:
    feature_length = _natural_feature_lengths_for_rows(rows)[-1]
    metadata = _metadata(feature_length)
    inputs_by_thread = [
        [
            _padded(
                feature_length,
                device=device,
                generator=generator,
            )
            for _ in range(iterations)
        ]
        for _ in range(2)
    ]
    with torch.inference_mode():
        expected_by_thread = [
            [
                run_audio_prefix_eager(
                    encoder,
                    padded,
                    *metadata,
                    async_tensor_h2d=async_tensor_h2d,
                )
                for padded in inputs
            ]
            for inputs in inputs_by_thread
        ]
    torch.cuda.synchronize(device)

    streams = [torch.cuda.Stream(device=device) for _ in range(2)]
    if streams[0].cuda_stream == streams[1].cuda_stream:
        raise AssertionError("concurrency gate requires distinct streams")
    barrier = threading.Barrier(2)

    def replay_many(
        thread_index: int,
        stream: torch.cuda.Stream,
    ) -> list[torch.Tensor]:
        torch.cuda.set_device(device)
        outputs = []
        try:
            with torch.inference_mode(), torch.cuda.stream(stream):
                for padded in inputs_by_thread[thread_index]:
                    barrier.wait()
                    outputs.append(
                        cache.run(
                            encoder,
                            padded,
                            *metadata,
                            async_tensor_h2d=async_tensor_h2d,
                        )
                    )
            stream.synchronize()
            return outputs
        except BaseException:
            barrier.abort()
            raise

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(replay_many, index, stream)
            for index, stream in enumerate(streams)
        ]
        outputs_by_thread = [future.result() for future in futures]

    for thread_index, (outputs, expected_outputs) in enumerate(
        zip(outputs_by_thread, expected_by_thread, strict=True)
    ):
        for iteration, (output, expected) in enumerate(
            zip(outputs, expected_outputs, strict=True)
        ):
            if not torch.equal(output, expected):
                raise AssertionError(
                    "concurrent prefix replay mismatch for "
                    f"thread={thread_index} iteration={iteration}"
                )
    print(
        "concurrency_gate=PASS threads=2 streams=2 "
        f"iterations={iterations} rows={rows}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows",
        action="append",
        type=int,
        help="Natural packed row count; repeat as needed. Defaults to 377..390.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--replay-checks", type=int, default=3)
    parser.add_argument("--concurrency-iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the prefix graph gate")
    if min(
        args.warmup,
        args.repeats,
        args.replay_checks,
        args.concurrency_iterations,
    ) <= 0:
        raise ValueError("all repeat and warmup counts must be positive")
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        raise RuntimeError("This production-shape gate requires an SM90 GPU")

    cases = args.rows or sorted(_NATURAL_PACKED_ROWS)
    if (
        len(cases) > len(_NATURAL_PACKED_ROWS)
        or len(set(cases)) != len(cases)
        or any(rows not in _NATURAL_PACKED_ROWS for rows in cases)
    ):
        raise ValueError("rows must be distinct values in the inclusive range 377..390")

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    torch.manual_seed(args.seed)
    encoder = PrefixOnlyEncoder().to(device=device).eval()
    cache = ExactShapeAudioPrefixGraphCache(
        max_entries=_MAX_CACHE_ENTRIES,
        warmup_iterations=args.warmup,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)

    print(
        "case,rows,primary_feature_len,alias_feature_len,chunks,stride,"
        "metadata_variants,bitwise_exact,kernel_order_exact,eager_kernels,"
        "graph_kernels,eager_cuda_us,replay_copy_graph_clone_cuda_us,"
        "cuda_speedup_pct,eager_host_us,replay_host_us,host_speedup_pct"
    )
    with torch.inference_mode():
        for case_index, rows in enumerate(cases):
            feature_lengths = _natural_feature_lengths_for_rows(rows)
            primary_feature_length = feature_lengths[-1]
            alias_feature_length = feature_lengths[0]
            primary_metadata = _metadata(primary_feature_length)
            alias_metadata = _metadata(alias_feature_length)
            primary_padded = _padded(
                primary_feature_length,
                device=device,
                generator=generator,
            )
            alias_padded = _padded(
                alias_feature_length,
                device=device,
                generator=generator,
            )

            _advance_exact_key_to_admission(
                cache,
                encoder,
                primary_padded,
                primary_metadata,
                expected_entry_delta=1,
                expected_admitted_delta=1,
            )
            _advance_exact_key_to_admission(
                cache,
                encoder,
                alias_padded,
                alias_metadata,
                expected_entry_delta=0,
                expected_admitted_delta=1,
            )

            retained_outputs = []
            retained_expected = []
            for _ in range(args.replay_checks):
                fresh_padded = _padded(
                    primary_feature_length,
                    device=device,
                    generator=generator,
                )
                retained_expected.append(
                    run_audio_prefix_eager(
                        encoder,
                        fresh_padded,
                        *primary_metadata,
                        async_tensor_h2d=async_tensor_h2d,
                    )
                )
                retained_outputs.append(
                    cache.run(
                        encoder,
                        fresh_padded,
                        *primary_metadata,
                        async_tensor_h2d=async_tensor_h2d,
                    )
                )
            alias_expected = run_audio_prefix_eager(
                encoder,
                alias_padded,
                *alias_metadata,
                async_tensor_h2d=async_tensor_h2d,
            )
            alias_output = cache.run(
                encoder,
                alias_padded,
                *alias_metadata,
                async_tensor_h2d=async_tensor_h2d,
            )
            torch.cuda.synchronize()
            bitwise_exact = all(
                torch.equal(output, expected)
                for output, expected in zip(
                    retained_outputs,
                    retained_expected,
                    strict=True,
                )
            ) and torch.equal(alias_output, alias_expected)
            if not bitwise_exact:
                raise AssertionError(f"prefix replay mismatch for rows={rows}")

            key = _make_prefix_graph_key(
                primary_padded,
                *primary_metadata,
            )
            if key is None:
                raise AssertionError("helper produced an unsupported exact key")
            signature = _make_prefix_graph_signature(key)
            entry = cache._entries[signature]
            device_pack_metadata = async_tensor_h2d(
                primary_metadata[1],
                dtype=torch.int32,
                device=device,
            )
            graph_body = lambda: run_audio_prefix_with_device_metadata(  # noqa: E731
                encoder,
                primary_padded,
                device_pack_metadata,
                total_rows=rows,
            )
            graph_body()
            entry.static_padded_feature.copy_(primary_padded)
            torch.cuda.synchronize()
            eager_kernels = _cuda_kernel_names(graph_body)
            graph_kernels = _cuda_kernel_names(
                lambda: cache._backend.replay(entry.graph)
            )
            kernel_order_exact = eager_kernels == graph_kernels
            if not kernel_order_exact:
                raise AssertionError(
                    "captured prefix kernel names/order differ from eager "
                    f"for rows={rows}: eager={eager_kernels}, graph={graph_kernels}"
                )

            eager_call = lambda: run_audio_prefix_eager(  # noqa: E731
                encoder,
                primary_padded,
                *primary_metadata,
                async_tensor_h2d=async_tensor_h2d,
            )
            replay_call = lambda: cache.run(  # noqa: E731
                encoder,
                primary_padded,
                *primary_metadata,
                async_tensor_h2d=async_tensor_h2d,
            )
            eager_cuda_us = _median_cuda_us(
                eager_call,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            replay_cuda_us = _median_cuda_us(
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
            cuda_speedup_pct = (eager_cuda_us / replay_cuda_us - 1) * 100
            host_speedup_pct = (eager_host_us / replay_host_us - 1) * 100
            print(
                f"{case_index},{rows},{primary_feature_length},"
                f"{alias_feature_length},{primary_padded.shape[0]},"
                f"{tuple(primary_padded.stride())},{len(feature_lengths)},"
                f"{bitwise_exact},{kernel_order_exact},{len(eager_kernels)},"
                f"{len(graph_kernels)},{eager_cuda_us:.3f},"
                f"{replay_cuda_us:.3f},{cuda_speedup_pct:.3f},"
                f"{eager_host_us:.3f},{replay_host_us:.3f},"
                f"{host_speedup_pct:.3f}"
            )

    if cache.entry_count != len(cases):
        raise AssertionError(
            f"expected {len(cases)} signatures, got {cache.entry_count}"
        )
    if cache.admitted_key_count != len(cases) * 2:
        raise AssertionError(
            "each row must admit both endpoint full-metadata variants"
        )
    concurrency_rows = 387 if 387 in cases else cases[0]
    _concurrent_same_key_gate(
        encoder,
        cache,
        rows=concurrency_rows,
        device=device,
        generator=generator,
        iterations=args.concurrency_iterations,
    )
    print(f"graph_signatures={cache.entry_count}")
    print(f"admitted_exact_keys={cache.admitted_key_count}")
    print(f"cuda_memory_allocated_mib={torch.cuda.memory_allocated() / 2**20:.3f}")
    print(f"cuda_memory_reserved_mib={torch.cuda.memory_reserved() / 2**20:.3f}")
    print("gate=PASS_EXACT_NATURAL_AUDIO_PREFIX_CUDAGRAPH")


if __name__ == "__main__":
    main()
