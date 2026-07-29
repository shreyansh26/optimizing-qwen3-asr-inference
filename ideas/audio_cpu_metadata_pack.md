# Audio encoder CPU metadata + valid-row pack experiment

## Hypothesis

The installed Qwen3-ASR audio encoder sends audio lengths to CUDA even though
they are derived on the CPU and used mostly for Python metadata. Its forward
then introduces device-to-host synchronization at all of these points:

- `chunk_num.sum()` while allocating chunk lengths;
- `chunk_lengths.tolist()` before splitting the input;
- `feature_lens_after_cnn.max().item()` while building the valid-row mask;
- boolean advanced indexing `padded_embed[padded_mask_after_cnn]`, whose dynamic
  output size requires a nonzero-count readback;
- `aftercnn_lens.tolist()` while constructing attention segments; and
- `audio_output_lengths.tolist()` while splitting final audio embeddings.

The preceding `audio_cpu_maxseqlen` experiment removes the repeated scalar
readback inside the 24 FlashAttention layers. This branch retains that patch and
targets the remaining metadata synchronization without changing convolution,
attention, or projection math.

## Seam and implementation

`MultiModalFieldConfig.batched("audio", keep_on_cpu=True)` is an intended vLLM
transport seam: its `BaseMultiModalField.reduce_data` explicitly keeps that
field on CPU when batching model inputs. The patch changes only
`audio_feature_lengths`; audio features and the feature attention mask preserve
their existing placement.

With lengths on CPU, the patched encoder forward:

1. Computes chunk lengths, post-CNN valid lengths, pack offsets, and cumulative
   attention offsets on CPU.
2. Runs the exact installed pad, convolution, positional embedding, transformer,
   and output-projection operations.
3. Replaces boolean advanced indexing with one Triton kernel. Each program copies
   one BF16 `[1024]` row, using CPU-computed valid lengths and output offsets sent
   to the GPU asynchronously.
4. Sends the already-cumulative attention offsets to CUDA, avoiding the original
   GPU `cumsum` as well as device-to-host metadata conversion.
5. Leaves whole-audio output lengths on CPU, so the model's final `.tolist()` is
   also host-local.

The packed tensor is a pure copy in the same chunk-major, row-major order as the
boolean-index reference. GPU validation confirmed that it is bit-exact before
the encoder transformer.

## Safety boundary

The experiment is enabled only when both variables are set:

```text
ASR_AUDIO_CPU_MAXSEQLEN=1
ASR_AUDIO_CPU_METADATA_PACK=1
```

Installation now requires vLLM `0.24.0+cu129` and checks that the installed
distribution's `direct_url.json` wheel URL matches a vLLM wheel URL in
`uv.lock` whose hash equals the pinned SHA-256. This is a declared
version-and-wheel-provenance guard; it does not hash installed Python function
source or installed wheel bytes. Runtime guards require the exact
Qwen3-ASR-1.7B FlashAttention encoder configuration, CUDA BF16
`[128, total_frames]` features, contiguous CPU integer lengths, inference mode,
and exact agreement between input lengths, output lengths, and feature storage.
Unsupported runtime inputs move the two length tensors back to the feature
device and call the untouched installed forward.

## Validation

CPU tests:

```bash
CUDA_VISIBLE_DEVICES='' \
UV_PROJECT_ENVIRONMENT=/mnt/ssd1/shreyansh/home_dir/asr_experiments/.venv \
/mnt/ssd1/shreyansh/home_dir/asr_experiments/.venv/bin/python -m unittest \
  tests.test_audio_cpu_maxseqlen_patch \
  tests.test_audio_cpu_metadata_pack_patch
```

The CUDA helper compares the packed output bit-for-bit, reports host-call and
CUDA timing, and lists profiler synchronization events for the installed-style
metadata path and candidate:

```bash
CUDA_VISIBLE_DEVICES=1 \
UV_PROJECT_ENVIRONMENT=/mnt/ssd1/shreyansh/home_dir/asr_experiments/.venv \
uv run inference/vllm_static_fp8/benchmarks/bench_audio_cpu_metadata_pack.py
```

Service launcher:

```bash
CUDA_VISIBLE_DEVICES=1 PORT=8091 \
  inference/experimental_launchers/run_vllm_fp8_static_qk_mrope_kv_cache_fusion_audio_cpu_metadata_pack.sh
```

### CUDA helper

The 448-chunk helper produced an exact `(5718, 1024)` output. The profiler found
three `cudaStreamSynchronize` events in the installed-style reference and none
in the candidate:

| Path | Host call | CUDA time | Stream synchronizations |
|---|---:|---:|---:|
| Installed-style reference | 156.688 us | 172.496 us | 3 |
| CPU metadata + Triton pack | 109.422 us | 123.424 us | 0 |

The candidate reduced helper host time by 30.2% and CUDA time by 28.4% while
preserving every BF16 output bit.

### Uniform 100 x 50-second batched benchmark

Six candidate runs reported:

```text
throughput: 22.502, 22.683, 22.367, 20.833, 22.332, 22.308 files/s
latency:     0.668,  0.660,  0.670,  0.719,  0.666,  0.663 s
TTFT:        0.176,  0.163,  0.178,  0.231,  0.194,  0.215 s
```

The time-adjacent CPU-max return control reported:

```text
throughput: 21.557, 21.194, 20.807 files/s
latency:     0.696,  0.710,  0.718 s
TTFT:        0.184,  0.206,  0.206 s
```

| Path | Runs | Throughput | Latency | TTFT |
|---|---:|---:|---:|---:|
| CPU-max return control | 3 | 21.186 files/s | 0.7080 s | 0.1987 s |
| CPU metadata + Triton pack | 6 | 22.1708 files/s | 0.6743 s | 0.1928 s |
| Candidate delta |  | **+4.65%** | **-4.76%** | **-2.94%** |

The return leg was noisy, including one visibly slower candidate run, so these
deltas use all six candidate runs and all three adjacent control runs rather
than selecting only favorable runs.

### Full 550-file batched benchmark

The candidate completed all 550 files. The comparison uses a fresh main control
and the full CPU-max experiment:

| Metric | Fresh main | CPU max | CPU metadata + pack | vs main | vs CPU max |
|---|---:|---:|---:|---:|---:|
| Throughput | 4.485 files/s | 4.619 files/s | **4.750 files/s** | **+5.91%** | **+2.84%** |
| Latency | 3.502 s | 3.397 s | **3.304 s** | **-5.65%** | **-2.74%** |
| TTFT | **0.474 s** | 0.528 s | 0.535 s | +12.87% | +1.33% |
| CER | 0.163900 | **0.160686** | 0.163614 | -0.000286 | +0.002928 |
| WER | 0.385612 | **0.382274** | 0.384638 | -0.000974 | +0.002364 |

The measured full run's local predictions are under
`inference/results_fp8_static_qk_audio_cpu_metadata_pack/`; the repeated
uniform runs used unique `/tmp/asr_audio_cpu_metadata_pack_*` roots. These
generated artifacts and their appended CSV rows are deliberately excluded from
the source commits. The README's reproduction commands use the repository's
normal `predictions/results_fp8_static_qk_audio_cpu_metadata_pack/` hierarchy
for new runs.

## Decision and final role

Accept this candidate as the general audio-metadata layer. It improves the prioritized full-set latency
by 5.65% over fresh main and 2.74% over CPU max, with corresponding throughput
gains. The full-set TTFT tradeoff is +12.87% versus main and +1.33% versus CPU
max, despite the uniform benchmark's small TTFT improvement. CER/WER do not show
a meaningful quality degradation: both are slightly better than fresh main,
and the small difference from the CPU-max run is within the observed run-level
quality variation. The exact low-level pack result and removal of all three
measured metadata synchronization events support promotion for the
latency-prioritized batched track.

This note's benchmark rows are the historical round-three evidence for that
layer, not the final endpoint. The promoted server now injects the natural-only
frontend and transformer CUDA graph runners into this same metadata-aware forward.
Graph misses continue through this CPU-metadata/Triton eager path; only a
metadata compatibility/runtime-guard miss returns to the original vLLM
forward. Current composed results are in
[`docs/audio-natural-only-cudagraph-benchmark.md`](../docs/audio-natural-only-cudagraph-benchmark.md).
