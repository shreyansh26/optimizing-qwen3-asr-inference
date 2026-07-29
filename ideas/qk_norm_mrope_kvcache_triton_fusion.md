# Q/K RMSNorm + MRoPE + KV-cache Triton fusion

## Status

Superseded intermediate candidate. It beats the static-FP8 control on batched
latency and throughput, but its direct 128-lane RMS reduction moves quality
more than necessary and its full-corpus TTFT regresses. The child
`opt2/qk-kvcache-exact-rms` branch retains the fusion while matching Inductor's
64-lane reduction order.

Branch: `opt2/qk-mrope-kvcache-fusion`

## Hypothesis

The Q/K RMSNorm+MRoPE fusion still leaves vLLM's
`reshape_and_cache_flash_kernel` as one launch per decoder layer. The K value
has already been produced inside the fused kernel, and V is a contiguous slice
of the QKV projection. The same Triton program can therefore scatter both into
the paged BF16 KV cache without another launch or another read of K.

## Implementation

The experiment extends `qk_mrope_kv_cache_fusion_patch.py` with a vLLM custom op that:

1. obtains the layer's KV cache and slot mapping through vLLM's forward
   context;
2. computes Q/K RMSNorm and interleaved three-axis MRoPE;
3. writes contiguous Q/K outputs;
4. scatters rotated K and the original V into the native BF16 paged cache; and
5. returns a dummy dependency consumed by `unified_attention_with_output`, so
   graph ordering is explicit.

The fast cache path is limited to the Qwen3-ASR-1.7B native layout
`[blocks, block_size, 8, 128]`. Unsupported or not-yet-initialized caches use a
correct separate-cache fallback. This fallback is required during vLLM's
pre-allocation memory-profile pass, where the cache is intentionally empty.

PyTorch 2.11.0+cu128 and vLLM 0.24.0 are unchanged.

## Correctness checks

Two isolated GPU checks compared the combined kernel with the prior Q/K fusion
plus vLLM's native `reshape_and_cache_flash`:

- contiguous slots spanning one cache block;
- shuffled slots spanning four blocks, including `-1` padding slots.

Both checks produced bit-identical Q, K, key-cache, and value-cache tensors.
A real 50-second ASR smoke transcript was byte-identical to all saved runs of
the Q/K-only fusion.

## Isolated CUDA-graph timing

The control here is the Q/K-only fusion followed by vLLM's native cache
scatter. Times are GPU CUDA-graph replay times.

| Tokens | Separate (us) | Combined (us) | Speedup |
|---:|---:|---:|---:|
| 1 | 4.603 | 2.827 | 1.63x |
| 16 | 5.010 | 3.073 | 1.63x |
| 128 | 5.759 | 3.348 | 1.72x |
| 512 | 8.834 | 5.027 | 1.76x |

## Batched benchmark screen

The candidate and unpatched static-FP8 control were alternated on the same
otherwise-free H100 (GPU 4). Each variant has six runs of 100 uniform
50-second inputs with 16 workers. Each server used its own compile cache.

| Variant, median of 6 | Throughput (files/s) | Avg latency (s) | Avg TTFT (s) |
|---|---:|---:|---:|
| Static-FP8 control | 22.107 | 0.675 | 0.128 |
| Combined fusion | 23.355 | 0.642 | 0.141 |
| Relative delta | +5.6% | -4.9% | +10.6% |

The means tell the same latency/throughput story: 21.692 to 22.798 files/s and
0.690 to 0.657 seconds latency. Mean TTFT moves from 0.142 to 0.153 seconds.
The host remains noisy, but both three-run candidate blocks beat their adjacent
three-run control block on median latency and throughput.

## Full 550-file batched result

| Variant | Files/s | Avg latency (s) | Avg TTFT (s) | CER | WER |
|---|---:|---:|---:|---:|---:|
| Paired static-FP8 control | 4.413 | 3.553 | 0.418 | 0.160771 | 0.381654 |
| Combined fusion | 4.548 | 3.451 | 0.475 | 0.163762 | 0.384453 |
| Delta | +3.1% | -2.9% | +57 ms | +0.002991 | +0.002799 |

This confirms the latency/throughput gain, but also shows why this arithmetic
version should not be selected: CER and WER each move by about 0.3 percentage
points and TTFT is worse. The exact-order child reduces those quality deltas by
more than an order of magnitude.

## Nsight Systems result

The dominant count-1 decode graph was compared over 20 replays.

| Metric per replay | Static FP8 | Q/K+MRoPE | Q/K+MRoPE+cache |
|---|---:|---:|---:|
| Kernel launches | 366 | 337 | 309 |
| Summed kernel time | 1.627 ms | 1.518 ms | 1.480 ms |
| Replay envelope | 1.649 ms | 1.537 ms | 1.610 ms |

The combined kernel takes 76.77 us/replay. The corresponding stock regions
take 106.18 us for Q/K norm+MRoPE and 72.44 us for cache scatter, saving about
101.85 us locally. Compared with the Q/K-only fusion, folding the cache write
saves 56.44 us of summed kernel time and another 28 launches.

The replay envelope does not retain the whole local saving: it is 2.4% below
the static-FP8 control but 4.7% above the Q/K-only fusion. The combined node
introduces more spacing between dependent graph kernels even while reducing
the total GPU work. This is the main follow-up tuning target.

Reports:

- static control: `inference/results/nsys/fp8static_c1_5s_node.sqlite`
- Q/K-only: `inference/results/nsys/qk_mrope_c1_5s_node.sqlite`
- combined: `inference/results/nsys/qk_mrope_kvcache_c1_5s_node.sqlite`

## Decision and next experiment

Keep the branch as the launch-fusion proof, but do not promote it as the final
setting because TTFT and quality regress.

A 50-second input produces a 660-token prompt, outside vLLM's default full
CUDA-graph capture limit of 512. Extended prefill capture was rejected on the
stock model, but it may interact differently with this Python-dispatched
custom op. Test exact 660/1320/1980 prompt capture on a child branch. Retain it
only if it recovers TTFT without erasing the latency/throughput gain.
