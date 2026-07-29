"""GPU correctness and host-latency check for the CPU max-seqlen fast path."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
import time

import torch

PATCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PATCH_DIR))

from audio_cpu_maxseqlen_patch import STATIC_MAX_SEQLEN  # noqa: E402


def _measure_host_item_us(value: torch.Tensor, repeats: int, layers: int) -> float:
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        for _ in range(layers):
            value.item()
        elapsed = time.perf_counter_ns() - start
        samples.append(elapsed / layers / 1_000)
    return statistics.median(samples)


def _run_case(lengths: list[int], heads: int, head_dim: int) -> float:
    from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
    from vllm.v1.attention.ops.vit_attn_wrappers import (
        flash_attn_maxseqlen_wrapper,
    )

    cu_cpu = torch.tensor([0, *torch.tensor(lengths).cumsum(0)], dtype=torch.int32)
    cu = cu_cpu.cuda()
    tokens = int(cu_cpu[-1])
    q, k, v = (
        torch.randn(
            1,
            tokens,
            heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for _ in range(3)
    )
    gpu_max = (cu[1:] - cu[:-1]).max()
    cpu_bound = torch.tensor(STATIC_MAX_SEQLEN, dtype=torch.int32)
    fa_version = get_flash_attn_version(head_size=head_dim)
    kwargs = dict(
        batch_size=1,
        is_rocm_aiter=False,
        fa_version=fa_version,
        scale=head_dim**-0.5,
        cu_seqlens=cu,
    )
    reference = flash_attn_maxseqlen_wrapper(q, k, v, max_seqlen=gpu_max, **kwargs)
    candidate = flash_attn_maxseqlen_wrapper(q, k, v, max_seqlen=cpu_bound, **kwargs)
    torch.cuda.synchronize()
    torch.testing.assert_close(candidate, reference, rtol=2e-2, atol=2e-2)
    return float((candidate.float() - reference.float()).abs().max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    cases = ([104, 103, 57, 13, 1], [63, 31, 7, 1], [13, 7, 1])
    for lengths in cases:
        max_abs = _run_case(list(lengths), args.heads, args.head_dim)
        print(f"lengths={list(lengths)} max_abs_diff={max_abs:.6g}")

    cu = torch.tensor([0, 104, 207, 264], dtype=torch.int32, device="cuda")
    gpu_max = (cu[1:] - cu[:-1]).max()
    cpu_bound = torch.tensor(STATIC_MAX_SEQLEN, dtype=torch.int32)
    gpu_us = _measure_host_item_us(gpu_max, args.repeats, args.layers)
    cpu_us = _measure_host_item_us(cpu_bound, args.repeats, args.layers)
    print(f"gpu_scalar_item_us_per_layer={gpu_us:.3f}")
    print(f"cpu_scalar_item_us_per_layer={cpu_us:.3f}")
    print(f"host_item_saved_us_per_24_layers={(gpu_us - cpu_us) * args.layers:.3f}")


if __name__ == "__main__":
    main()
