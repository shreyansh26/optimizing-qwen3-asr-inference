# Final natural-only audio CUDA graphs

Branch: `promote/audio-prefix-shared-suffix-bucketed`

Date: 2026-07-21

## Decision

The promoted server composes four optimizations:

1. calibrated static FP8 decoder execution;
2. fused Q/K RMSNorm, MRoPE, and paged KV-cache writes;
3. CPU-built audio length/attention metadata plus a Triton valid-row pack;
4. exact-admitted CUDA graph caches for the audio frontend and transformer.

The CUDA graph policy is deliberately **natural-only**. It captures canonical
29--30 second chunks created by the server splitter and does not capture the
approximately 20--21 second tails induced by the fixed-50 benchmark. Arbitrary
durations remain supported by the general CPU-metadata/eager path.

Launch the final stack with:

```bash
PORT=8091 \
  bash inference/run_vllm_fp8_static_audio_encoder_cudagraphs.sh
```

For the exact audio-length mapping and implementation flow, see
[Qwen3-ASR audio lengths and CUDA-graph fast-path coverage](qwen3-asr-audio-length-and-graph-fast-path.md).

## What is cached

The graph boundary follows two named audio-encoder stages:

```text
input features
  -> chunk/pad -> three CNNs -> conv_out -> position add -> valid-row pack
     <----------------------- frontend graph ------------------------------>
  -> 24 audio-transformer layers -> ln_post -> proj1 -> act -> proj2
     <----------------------- transformer graph ------------------------------>
```

The tensor at the boundary is `[M, 1024]`, so the same admitted post-CNN row
range `M=377..390` appears in both caches.

- The audio frontend accepts 104 exact feature lengths `F=2897..3000`. They collapse
  to 14 captured signatures, one for each output row count `M=377..390`.
- All frontend signatures use one CUDA graph memory pool. Capture/replay is
  serialized because pool-backed allocations may alias across signatures.
- The audio transformer accepts the same 14 exact row counts but pads them into one fixed
  390-row graph. The returned tensor is sliced back to the real `M` rows.
- Exact input metadata is validated before either cache can select a graph;
  equal total rows alone are insufficient.

Each exact runtime key has an eight-observation probation period:

```text
observations 1..7 -> eager execution
observation 8     -> capture or signature alias, then bitwise eager validation
observation 9+    -> copy into stable inputs, replay, clone/slice output
```

Unsupported keys, cache limits, capture failures, validation mismatches,
training/gradient mode, and incompatible tensor layouts fail closed to eager
execution. The caches do not change model outputs approximately: a candidate
must match eager output bitwise before hot replay is admitted.

## What remains general

The CPU metadata layer is used for every supported audio length, not only graph
shapes. It keeps feature lengths on the CPU, derives raw chunk lengths,
post-CNN pack lengths and offsets, and local-attention cumulative boundaries,
then transfers only the small metadata tensors asynchronously. One Triton
kernel packs valid CNN rows directly into `[M, 1024]`. A companion max-seqlen
patch keeps the known attention maximum on the CPU instead of repeating a GPU
scalar readback in each of the 24 transformer layers.

Therefore an audio request has three possible outcomes:

| Runtime shape/layout | Metadata and row pack | Audio-encoder execution |
|---|---|---|
| Canonical `F=2897..3000`, admitted and hot | CPU + Triton | CUDA graph replay |
| CPU-metadata-compatible but graph-ineligible length/metadata | CPU + Triton | eager |
| Unsupported installed model/runtime contract | original vLLM path | original vLLM path |

## Benchmark provenance

The tables below are the output selected from the current local CSV files by:

```bash
uv run inference/analyse_results.py --mode batched
uv run inference/analyse_results.py --mode sequential
```

The fixed-50 final rows used 100 warm-up files. Most older fixed-50 controls
printed beside them used 20, so their percentage deltas are useful snapshot
comparisons rather than a warm-up-matched causal A/B. The full 550-file rows
used 20 warm-up files and are the only rows scored for CER/WER, but were also
collected on different dates/server lifetimes. All final service runs completed
without a failed or timed-out measured request. Latency and TTFT are seconds;
throughput is completed files per second.

## Batched results

### Full workload, 550 files, 16 workers

| Precision | Lat p50 | Lat p95 | Lat p99 | TTFT p50 | TTFT p95 | TTFT p99 | Throughput | CER | WER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 3.520 | 7.136 | 9.130 | 0.351 | 0.946 | 1.946 | 4.138 | 0.166 | 0.387 |
| FP8 dynamic | 3.563 | 7.166 | 9.903 | 0.343 | 1.121 | 1.838 | 4.104 | 0.158 | 0.380 |
| FP8 static | 3.346 | 6.675 | 9.124 | 0.426 | 1.177 | 1.984 | 4.352 | 0.162 | 0.384 |
| FP8 static + Q/K fusion | 3.137 | 6.276 | 8.382 | 0.385 | 1.069 | 1.761 | 4.604 | 0.166 | 0.387 |
| **Final natural-only graphs** | **2.521** | **4.830** | **6.653** | 0.543 | 1.328 | **1.759** | **5.722** | **0.160** | **0.382** |

Versus the displayed fused static-FP8 decoder snapshot:

| Metric | Change |
|---|---:|
| Throughput | **+24.28%** |
| Latency p50 / p95 / p99 | **-19.64% / -23.04% / -20.63%** |
| TTFT p50 / p95 / p99 | +41.04% / +24.23% / **-0.11%** |

This is the primary latency-prioritized result. TTFT p50/p95 remain the main
tradeoff under the varying 16-worker load.

### Fixed 50 seconds, 100 files, 16 workers

| Precision | Lat p50 | Lat p95 | Lat p99 | TTFT p50 | TTFT p95 | TTFT p99 | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 0.720 | 1.338 | 2.768 | 0.138 | 0.448 | 0.456 | 18.554 |
| FP8 dynamic | 0.735 | 1.279 | 1.481 | 0.154 | 0.432 | 0.499 | 20.731 |
| FP8 static | 0.652 | 1.098 | 1.212 | 0.119 | 0.306 | 0.332 | 22.171 |
| FP8 static + Q/K fusion | 0.636 | 1.226 | 1.424 | 0.122 | 0.499 | 0.614 | 22.635 |
| **Final natural-only graphs** | **0.426** | **0.890** | **1.008** | **0.086** | **0.216** | **0.227** | **28.192** |

Versus the displayed fused-control snapshot, throughput improves `24.55%`, latency percentiles
improve `27.41--33.02%`, and TTFT percentiles improve `29.51--63.03%`.
Only the first approximately 29--30 second chunk is graph eligible; the final
approximately 20--21 second tail is eager. The benefit therefore does not
depend on retaining benchmark-specific tail graphs.

## Sequential results

### Full workload, 550 files

| Precision | Lat p50 | Lat p95 | Lat p99 | TTFT p50 | TTFT p95 | TTFT p99 | Throughput | CER | WER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 1.576 | 3.261 | 4.229 | 0.203 | 0.318 | 0.377 | 0.583 | 0.168 | 0.388 |
| FP8 dynamic | 1.473 | 2.929 | 3.732 | 0.201 | 0.309 | 0.339 | 0.640 | 0.165 | 0.385 |
| FP8 static | 1.381 | 2.745 | 3.439 | 0.207 | 0.339 | 0.372 | 0.682 | 0.162 | 0.384 |
| FP8 static + Q/K fusion | 1.315 | 2.599 | 3.275 | 0.220 | 0.341 | 0.373 | 0.716 | 0.163 | 0.385 |
| **Final natural-only graphs** | **1.214** | **2.494** | **3.110** | **0.203** | **0.301** | **0.329** | **0.763** | **0.160** | **0.383** |

Versus the displayed fused-control snapshot, throughput improves `6.56%`, latency percentiles
improve `4.04--7.68%`, and TTFT percentiles improve `7.73--11.80%`.

### Fixed 50 seconds, 100 files

| Precision | Lat p50 | Lat p95 | Lat p99 | TTFT p50 | TTFT p95 | TTFT p99 | Throughput |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 0.365 | 0.564 | 1.411 | 0.063 | 0.074 | 0.092 | 2.677 |
| FP8 dynamic | 0.286 | 0.479 | 0.634 | 0.046 | 0.060 | 0.064 | 3.409 |
| FP8 static | 0.273 | 0.468 | 0.489 | 0.059 | 0.075 | 0.090 | 3.538 |
| FP8 static + Q/K fusion | 0.262 | 0.451 | 0.473 | 0.063 | 0.087 | 0.091 | 3.731 |
| **Final natural-only graphs** | **0.202** | **0.333** | **0.398** | **0.050** | **0.063** | **0.077** | **4.551** |

Versus the displayed fused-control snapshot, throughput improves `21.98%`, latency percentiles
improve `15.86--26.16%`, and TTFT percentiles improve `15.38--27.59%`.

## Quality interpretation

The final full rows report CER/WER of `0.160/0.383` sequentially and
`0.160/0.382` in batched mode. The fused decoder rows are `0.163/0.385` and
`0.166/0.387`, respectively. These snapshots show no material CER/WER
degradation. They are service-level decoding results rather than proof of
transcript determinism; the lower-level graph gate separately requires
bitwise equality against eager encoder output before admitting replay.

## Generality and limits

The result is **shape-specialized but workload-general**:

- audio content, language, speaker, file name, codec, and original sample rate
  are not graph keys;
- decoded/resampled feature length and the complete packing/attention layout
  determine eligibility;
- every 29--30 second non-final chunk generated by the current splitter can
  qualify independently;
- the last chunk of a long file may be anywhere in `(0, 30]` seconds and is
  eager unless it independently lies in the natural range;
- a multi-audio layout cannot qualify only because its total row count matches;
- arbitrary lengths continue to benefit from CPU metadata and exact Triton
  row packing even when neither CUDA graph is used.

The 550-file gain therefore reflects useful partial coverage across varying
audio durations, not specialization to the identities or exact lengths of the
benchmark files.

## Validation commands

```bash
UV_PROJECT_ENVIRONMENT=/mnt/ssd1/shreyansh/home_dir/asr_experiments/.venv \
uv run python -m unittest \
  tests.test_audio_cpu_metadata_pack_patch \
  tests.test_audio_cpu_maxseqlen_patch \
  tests.test_audio_frontend_cudagraph_patch \
  tests.test_audio_transformer_cudagraph_patch \
  tests.test_audio_encoder_cudagraph_patch

uv run inference/analyse_results.py --mode batched
uv run inference/analyse_results.py --mode sequential
```
