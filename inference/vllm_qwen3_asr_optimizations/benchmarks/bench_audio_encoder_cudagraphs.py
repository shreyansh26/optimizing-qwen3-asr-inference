#!/usr/bin/env python3
"""CUDA gate for the chained Qwen3-ASR audio-encoder graphs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading

import torch

PATCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PATCH_DIR))

from audio_cpu_metadata_pack_patch import (
    _build_cpu_metadata,
    run_audio_frontend_eager,
    run_audio_transformer_eager,
)  # noqa: E402
from audio_frontend_cudagraph_patch import (
    _NATURAL_PACKED_ROWS,
    ExactShapeAudioFrontendGraphCache,
)  # noqa: E402
from audio_transformer_cudagraph_patch import (  # noqa: E402
    ExactShapeAudioTransformerGraphCache,
)
from bench_audio_frontend_cudagraph import (
    _median_cuda_us,
    _median_host_call_us,
)  # noqa: E402
from bench_audio_transformer_cudagraph import (
    _initialize_single_gpu_distributed,
    _new_encoder,
)  # noqa: E402
from vllm.utils.torch_utils import async_tensor_h2d  # noqa: E402


_REPRESENTATIVE_FEATURE_LENGTHS = {
    377: 2900,
    **{rows: 2900 + (rows - 377) * 8 for rows in range(378, 390)},
    390: 3000,
}


def _natural_metadata(total_rows: int) -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    feature_length = _REPRESENTATIVE_FEATURE_LENGTHS[total_rows]
    chunk_lengths, pack_metadata, cu_seqlens = _build_cpu_metadata(
        torch.tensor([feature_length], dtype=torch.int64),
        torch.tensor([total_rows], dtype=torch.int64),
        n_window=50,
        n_window_infer=800,
    )
    return (
        chunk_lengths,
        pack_metadata,
        tuple(cu_seqlens),
        (feature_length,),
        (total_rows,),
    )


def _natural_padded(
    chunk_lengths: torch.Tensor,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    chunks = [
        torch.randn(
            (int(length), 128),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        for length in chunk_lengths.tolist()
    ]
    return torch.nn.utils.rnn.pad_sequence(
        chunks,
        batch_first=True,
    ).transpose(1, 2).unsqueeze(1)


def _concurrent_chain_gate(
    encoder,
    frontend_cache,
    transformer_cache,
    metadata,
    cu_seqlens,
    max_seqlen,
    *,
    device: torch.device,
    generator: torch.Generator,
    iterations: int,
) -> None:
    inputs_by_thread = [
        [
            _natural_padded(
                metadata[0],
                device=device,
                generator=generator,
            )
            for _ in range(iterations)
        ]
        for _ in range(2)
    ]

    def eager(padded_feature):
        hidden_states = run_audio_frontend_eager(
            encoder,
            padded_feature,
            *metadata,
            async_tensor_h2d=async_tensor_h2d,
        )
        return run_audio_transformer_eager(
            encoder,
            hidden_states,
            cu_seqlens,
            max_seqlen,
            cu_seqlens_values=metadata[2],
        )

    with torch.inference_mode():
        expected_by_thread = [
            [eager(padded_feature) for padded_feature in thread_inputs]
            for thread_inputs in inputs_by_thread
        ]
    torch.cuda.synchronize(device)

    streams = [torch.cuda.Stream(device=device) for _ in range(2)]
    barrier = threading.Barrier(2)

    def replay_many(thread_index: int) -> list[torch.Tensor]:
        stream = streams[thread_index]
        torch.cuda.set_device(device)
        outputs = []
        try:
            with torch.inference_mode(), torch.cuda.stream(stream):
                for padded_feature in inputs_by_thread[thread_index]:
                    barrier.wait()
                    hidden_states = frontend_cache.run(
                        encoder,
                        padded_feature,
                        *metadata,
                        async_tensor_h2d=async_tensor_h2d,
                    )
                    outputs.append(
                        transformer_cache.run(
                            encoder,
                            hidden_states,
                            cu_seqlens,
                            max_seqlen,
                            cu_seqlens_values=metadata[2],
                        )
                    )
            stream.synchronize()
            return outputs
        except BaseException:
            barrier.abort()
            raise

    with ThreadPoolExecutor(max_workers=2) as executor:
        outputs_by_thread = list(executor.map(replay_many, (0, 1)))

    for thread_index, (outputs, expected_outputs) in enumerate(
        zip(outputs_by_thread, expected_by_thread, strict=True)
    ):
        for iteration, (output, expected) in enumerate(
            zip(outputs, expected_outputs, strict=True)
        ):
            if not torch.equal(output, expected):
                raise AssertionError(
                    "combined replay is not bitwise exact for "
                    f"thread={thread_index}, iteration={iteration}"
                )


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=384)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--concurrency-iterations", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=29619)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.rows not in _NATURAL_PACKED_ROWS:
        raise ValueError("rows must be one of the admitted natural chunk sizes")
    if args.warmup <= 0 or args.repeats <= 0 or args.concurrency_iterations <= 0:
        raise ValueError(
            "--warmup, --repeats, and --concurrency-iterations must be positive"
        )
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        raise RuntimeError("This production-shape gate requires an SM90 GPU")

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    _initialize_single_gpu_distributed(args.master_port)
    encoder = _new_encoder(device)
    metadata = _natural_metadata(args.rows)
    cu_seqlens = torch.tensor(metadata[2], dtype=torch.int32, device=device)
    max_seqlen = torch.tensor(104, dtype=torch.int32, device="cpu")
    frontend_cache = ExactShapeAudioFrontendGraphCache(
        warmup_iterations=args.warmup,
    )
    transformer_cache = ExactShapeAudioTransformerGraphCache(
        warmup_iterations=args.warmup,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    padded = _natural_padded(
        metadata[0],
        device=device,
        generator=generator,
    )

    def eager(input_tensor=padded):
        hidden_states = run_audio_frontend_eager(
            encoder,
            input_tensor,
            *metadata,
            async_tensor_h2d=async_tensor_h2d,
        )
        return run_audio_transformer_eager(
            encoder,
            hidden_states,
            cu_seqlens,
            max_seqlen,
            cu_seqlens_values=metadata[2],
        )

    def candidate(input_tensor=padded):
        hidden_states = frontend_cache.run(
            encoder,
            input_tensor,
            *metadata,
            async_tensor_h2d=async_tensor_h2d,
        )
        return transformer_cache.run(
            encoder,
            hidden_states,
            cu_seqlens,
            max_seqlen,
            cu_seqlens_values=metadata[2],
        )

    with torch.inference_mode():
        eager_output = eager()
        for _ in range(7):
            probation_output = candidate()
            torch.cuda.synchronize(device)
            if not torch.equal(probation_output, eager_output):
                raise AssertionError("combined probation output is not exact")
            if frontend_cache.entry_count != 0:
                raise AssertionError("frontend graph captured before observation 8")

        eighth_output = candidate()
        torch.cuda.synchronize(device)
        if not torch.equal(eighth_output, eager_output):
            raise AssertionError("combined observation-8 output is not exact")
        if frontend_cache.entry_count != 1 or transformer_cache.entry_count != 1:
            raise AssertionError("both graph caches must be admitted by call 8")

        other_padded = _natural_padded(
            metadata[0],
            device=device,
            generator=generator,
        )
        other_eager = eager(other_padded)
        other_candidate = candidate(other_padded)
        torch.cuda.synchronize(device)
        if not torch.equal(other_candidate, other_eager):
            raise AssertionError("different-content combined replay is not exact")

        _concurrent_chain_gate(
            encoder,
            frontend_cache,
            transformer_cache,
            metadata,
            cu_seqlens,
            max_seqlen,
            device=device,
            generator=generator,
            iterations=args.concurrency_iterations,
        )

        for _ in range(args.warmup):
            eager()
            candidate()
        torch.cuda.synchronize(device)

        print("correctness=bitwise_exact")
        print("probation=PASS observations=8")
        print("concurrency=PASS threads=2 streams=2")
        print(f"rows={args.rows}")
        print(
            f"eager_host_us={_median_host_call_us(eager, repeats=args.repeats):.3f}"
        )
        print(
            "combined_host_us="
            f"{_median_host_call_us(candidate, repeats=args.repeats):.3f}"
        )
        print(
            "eager_cuda_us="
            f"{_median_cuda_us(eager, warmup=args.warmup, repeats=args.repeats):.3f}"
        )
        candidate_us = _median_cuda_us(
            candidate,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        print(f"combined_copy_replay_clone_cuda_us={candidate_us:.3f}")
        print("gate=PASS_EXACT_AUDIO_ENCODER_CUDAGRAPHS")


def main() -> None:
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
