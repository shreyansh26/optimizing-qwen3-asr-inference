"""CUDA correctness, synchronization, and timing check for audio CPU metadata."""

from __future__ import annotations

import argparse
from itertools import accumulate
from pathlib import Path
import statistics
import sys
import time
from typing import Callable

import torch

PATCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PATCH_DIR))

from audio_cpu_metadata_pack_patch import pack_valid_rows  # noqa: E402
from vllm.utils.torch_utils import async_tensor_h2d  # noqa: E402


def _metadata(chunks: int, rows: int) -> tuple[torch.Tensor, list[int]]:
    lengths = [rows] * chunks
    for index in range(min(16, chunks)):
        lengths[chunks - index - 1] = 1 + (index * 7) % rows
    offsets = [0, *accumulate(lengths)]
    pack = torch.tensor(lengths + offsets[:-1], dtype=torch.int32)
    return pack, offsets


def _host_call_us(function: Callable[[], torch.Tensor], repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        output = function()
        samples.append((time.perf_counter_ns() - start) / 1_000)
        del output
    torch.cuda.synchronize()
    return statistics.median(samples)


def _cuda_call_us(function: Callable[[], torch.Tensor], repeats: int) -> float:
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


def _sync_events(function: Callable[[], torch.Tensor]) -> dict[str, int]:
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        output = function()
    del output
    return {
        event.key: event.count
        for event in profile.key_averages()
        if "Synchronize" in event.key
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=448)
    parser.add_argument("--rows", type=int, default=13)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.rows <= 0 or args.chunks <= 0 or args.hidden_size != 1024:
        raise ValueError("Expected positive chunks/rows and hidden size 1024")

    padded = torch.randn(
        (args.chunks, args.rows, args.hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    pack_metadata, offsets = _metadata(args.chunks, args.rows)
    lengths_gpu = async_tensor_h2d(
        pack_metadata[: args.chunks],
        dtype=torch.int32,
        device=padded.device,
    )
    aftercnn_cpu = torch.tensor(
        [offsets[-1]],
        dtype=torch.int64,
        device="cpu",
    )
    aftercnn_gpu = aftercnn_cpu.to(device=padded.device)

    def reference() -> torch.Tensor:
        max_rows = lengths_gpu.max().item()
        indices = torch.arange(max_rows, device=padded.device)
        mask = indices.unsqueeze(0) < lengths_gpu.unsqueeze(1)
        output = padded[mask]
        segment_lengths = []
        for length in aftercnn_gpu.tolist():
            segment_lengths.append(int(length))
        async_tensor_h2d(
            [0, *accumulate(segment_lengths)],
            dtype=torch.int32,
            device=padded.device,
        ).cumsum(-1, dtype=torch.int32)
        return output

    def candidate() -> torch.Tensor:
        output = pack_valid_rows(
            padded,
            pack_metadata,
            async_tensor_h2d=async_tensor_h2d,
        )
        segment_lengths = [int(length) for length in aftercnn_cpu.tolist()]
        async_tensor_h2d(
            [0, *accumulate(segment_lengths)],
            dtype=torch.int32,
            device=padded.device,
        )
        return output

    # Compile the Triton specialization before collecting timing/profiler data.
    for function in (reference, candidate):
        for _ in range(3):
            function()
    torch.cuda.synchronize()

    reference_output = reference()
    candidate_output = candidate()
    torch.cuda.synchronize()
    torch.testing.assert_close(candidate_output, reference_output, rtol=0, atol=0)
    print(
        "correctness=exact",
        f"output_shape={tuple(candidate_output.shape)}",
    )

    for name, function in (
        ("reference", reference),
        ("candidate", candidate),
    ):
        print(
            f"{name}_host_call_us={_host_call_us(function, args.repeats):.3f}"
        )
        print(f"{name}_cuda_us={_cuda_call_us(function, args.repeats):.3f}")
        print(f"{name}_profiler_sync_events={_sync_events(function)}")


if __name__ == "__main__":
    main()
