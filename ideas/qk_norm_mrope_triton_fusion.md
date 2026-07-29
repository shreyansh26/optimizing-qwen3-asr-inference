# Q/K RMSNorm + MRoPE Triton fusion

## Status

Profile-level success, but not an end-to-end benchmark winner on the first
screen. Keep this experiment isolated until it can be combined with another
decode-path saving.

## Hypothesis

Every Qwen3 decoder layer launches one reduction kernel for Q/K RMSNorm and a
second pointwise kernel for three-axis MRoPE. A specialized Triton kernel can
perform both operations in one launch while keeping the normalized values in
FP32 through the rotation, matching the numerical structure of the compiled
control graph.

The implementation is deliberately limited to the Qwen3-ASR-1.7B layout:

- 16 query heads and 8 KV heads
- head dimension 128
- BF16 Q/K projection outputs
- interleaved MRoPE sections `[24, 20, 20]`

Unsupported layouts fall back to the original vLLM attention forward.

## Implementation

- `inference/vllm_static_fp8/qk_mrope_kv_cache_fusion_patch.py` patches
  `Qwen3Attention.forward` at plugin load time.
- One Triton program handles 16 heads, computes the per-head RMS reduction,
  applies the learned RMSNorm weight, selects the temporal/height/width
  position for each frequency, and applies MRoPE.
- `ASR_QK_MROPE_KV_CACHE_FUSION=0` disables the patch. The static-FP8 launch script
  enables it by default on this branch.
- PyTorch and vLLM versions are unchanged.

## Isolated kernel result

CUDA-graph replay timing was measured against the two kernels emitted by the
compiled control graph. The fused kernel is consistently faster over decode
and representative prefill token counts:

| Tokens | Compiled control (us) | Fused (us) |
|---:|---:|---:|
| 32 | 8.604 | 2.583 |
| 128 | 8.781 | 2.930 |
| 512 | 11.048 | 4.301 |
| 2,048 | 33.694 | 10.441 |
| 8,192 | 117.165 | 39.245 |

The plain Python launch path is not representative because its CPU launch
overhead dominates; the table uses CUDA-graph replay.

## Nsight Systems result

Both reports profile the server process during the same count-1 workload. The
dominant decoded CUDA graph was compared over 20 replays.

| Metric per replay | Static-FP8 control | Q/K+MRoPE fusion | Delta |
|---|---:|---:|---:|
| Kernel launches | 366 | 337 | -29 |
| Sum of kernel durations | 1.627 ms | 1.518 ms | -6.7% |
| Replay envelope | 1.649 ms | 1.537 ms | -6.8% |
| Q/K norm + MRoPE region | 106.18 us | 58.47 us | -44.9% |

Reports:

- control: `inference/results/nsys/fp8static_c1_5s_node.sqlite`
- candidate: `inference/results/nsys/qk_mrope_c1_5s_node.sqlite`

The reduction in launches is the expected one fused launch per decoder layer,
plus a graph-name/layout difference. The replay result is strong evidence that
the fusion works even though host-level benchmark variance masks it.

## Batched screen

The screen used one server replica on GPU 2, 16 benchmark workers, and 100
uniform 50-second inputs. The candidate used a fresh `VLLM_CACHE_ROOT`, so it
did not reuse the control's compiled graph.

| Variant | Throughput (files/s) | Avg latency (s) | Avg TTFT (s) |
|---|---:|---:|---:|
| Candidate median, first 3 | 22.286 | 0.669 | 0.140 |
| Interleaved control median, 3 | 22.528 | 0.659 | 0.119 |

Individual candidate runs ranged from 20.581 to 22.799 files/s and individual
control runs ranged from 20.373 to 23.134 files/s. This host-level variance is
larger than the expected gain, so the candidate cannot be called faster from
`run_benchmark.py` yet. Latency and TTFT are slightly worse at the medians.

## Accuracy proxy

The 100 clipped uniform inputs are not the corpus-quality gate, but a direct
reference-relative comparison showed only small movement:

| Variant | Proxy CER | Proxy WER |
|---|---:|---:|
| Control | 0.842183 | 0.889981 |
| Candidate | 0.842643 | 0.890358 |
| Absolute delta | +0.000460 | +0.000377 |

Fifty-four of 100 transcripts differed, which is expected from the different
floating-point reduction order. Full-corpus CER/WER should only be run if a
combined candidate wins the performance screen.

## Decision and follow-up

Do not merge this branch alone: it has a real 6.8% graph-replay reduction but
does not yet beat the static-FP8 control on end-to-end batched latency. The most
promising combinations are:

1. fold KV-cache reshape/scatter into the same Q/K path, eliminating another
   per-layer launch;
2. add a static-FP8 output epilogue to the existing FlashAttention-3 path
   without forcing FlashAttention-4 or split-K;
3. tune the dominant CUTLASS FP8 GEMM configurations, then retain this fusion
   if the combined improvement clears host variance.
