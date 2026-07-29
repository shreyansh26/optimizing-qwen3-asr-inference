# Qwen3-ASR audio lengths and CUDA-graph fast-path coverage

This note explains how decoded audio duration becomes a Qwen3-ASR feature
length, how that feature length becomes the post-CNN row count used by the
audio transformer, and which resulting shapes are admitted by the current
audio-frontend and audio-transformer CUDA graphs.

The implementation described here is the final path on branch
`promote/audio-prefix-shared-suffix-bucketed`. Earlier branches first added a
shared prefix pool and two suffix buckets, then experimentally expanded a
21-chunk tail family to `M=263..273`. The promoted implementation deliberately
removes that workload-derived tail admission and retains graphs only for the
canonical 29--30 second chunk family. It combines:

- the accepted static-FP8 decoder with fused Q/K RMSNorm, MRoPE, and KV-cache
  write;
- general CPU construction of audio length and packing metadata;
- exact-admitted audio-frontend CUDA graphs with a shared graph memory pool;
- exact-admitted audio-transformer CUDA graphs grouped into one padded 390-row bucket.

The key distinction is:

> The optimization does not recognize an audio file, speaker, language, or
> waveform. It recognizes exact tensor shapes and metadata after decoding and
> resampling. An unseen audio file can use the graph fast path when it maps to
> an admitted shape.

This documentation uses stage names rather than the earlier positional names:

- **audio frontend**: the three CNNs, `conv_out`, positional embedding, and
  valid-row pack, producing `[M, 1024]`;
- **audio transformer**: the 24 transformer layers, `ln_post`, and output
  projections, producing `[M, 2048]`;
- **audio encoder CUDA graphs**: the frontend and transformer caches together.

The retired “prefix” and “suffix” terms only meant “before” and “after” the
packed `[M, 1024]` boundary. They did not describe audio time, prompt
placement, or vLLM's text prefix cache.

## End-to-end length pipeline

For one server-side audio chunk, the relevant pipeline is:

```text
encoded audio file
  -> decode, channel normalization, and resampling to 16 kHz
  -> valid 128-bin log-mel frames, feature length F
  -> split F into 100-frame convolution chunks
  -> three stride-two convolutions per chunk
  -> pack valid post-CNN rows, total length M
  -> divide M into local-attention sequences of at most 104 rows
  -> audio transformer and projection
```

Files longer than 30 seconds are split into several server-side chunks before
this model pipeline. Graph admission happens independently for each resulting
chunk, not once for the original file.

## Final promoted execution flow

The launcher enables all four audio flags before chaining into the fused
static-FP8 server:

```text
run_vllm_fp8_static_audio_encoder_cudagraphs.sh
  -> ASR_AUDIO_CPU_MAXSEQLEN=1
  -> ASR_AUDIO_CPU_METADATA_PACK=1
  -> ASR_AUDIO_FRONTEND_CUDAGRAPH=1
  -> ASR_AUDIO_TRANSFORMER_CUDAGRAPH=1
  -> run_vllm_fp8_static_qk_mrope_kv_cache_fusion.sh
```

`audio_encoder_cudagraph_patch.py` installs both graph runners through a
single patched audio-encoder forward owned by
`audio_cpu_metadata_pack_patch.py`. For each encoder invocation:

```text
CPU feature_lens and aftercnn_lens
  -> validate installed model, backend, tensors, and exact length formula
  -> derive chunk_lengths, pack lengths/offsets, and cu_seqlens on CPU
  -> split and pad GPU input features using CPU-known chunk lengths
  -> audio frontend runner
       admitted hot natural key -> frontend CUDA graph
       otherwise                -> eager CNN/projection + Triton row pack
  -> async copy cu_seqlens to GPU; obtain CPU-cached max_seqlen
  -> audio transformer runner
       admitted hot natural key -> padded 390-row transformer CUDA graph
       otherwise                -> eager 24-layer transformer/projection
```

The CUDA graphs are therefore accelerators inside the general metadata path,
not an alternative input pipeline. A request can miss both graph caches and
still receive the CPU-metadata and Triton-pack optimization.

## Decoded samples to feature length

Let `N` be the number of mono samples after resampling to 16 kHz. The Whisper
feature extractor uses a hop length of 160 samples, which is 10 ms at 16 kHz.

The installed vLLM multimodal processor first right-pads each positive decoded
chunk to a multiple of the 160-sample hop. It then publishes the padded sample
count divided by 160 as `audio_feature_lengths`. Therefore the runtime feature
length is:

```text
F = ceil(N / 160) = floor((N + 159) / 160)
```

Equivalently, for decoded duration `T = N / 16000` seconds:

```text
F = ceil(100 * T)
```

The exact runtime authority is the patched `audio_feature_lengths` and its
matching feature attention mask. Thus an exact positive feature length `F`
corresponds to the following 16 kHz sample interval:

```text
160 * (F - 1) < N <= 160 * F
```

or the following half-open/half-closed duration interval:

```text
(F - 1) / 100 < T <= F / 100 seconds
```

Original sample rate, channel count, and container format do not directly
select a graph. They matter only insofar as decoding and resampling change the
final 16 kHz sample count. For compressed formats or resampled inputs, inspect
the decoded sample count or runtime feature attention mask when a boundary
must be classified exactly.

The installed pre-padding and length override are in:

```text
.venv/lib/python3.12/site-packages/vllm/model_executor/models/
  qwen3_omni_moe_thinker.py
```

The underlying feature extractor is in:

```text
.venv/lib/python3.12/site-packages/transformers/models/whisper/
  feature_extraction_whisper.py
```

## Feature length to post-CNN rows

The audio encoder first divides `F` into raw chunks of 100 feature frames:

```text
q, r = divmod(F, 100)
```

Here `q` is the number of full chunks and `r` is the final partial chunk
length. One 100-frame chunk represents approximately one second.

Each raw chunk passes through three convolutions with stride two and padding
one. For a chunk with `L` valid frames, each convolution applies:

```text
L -> floor((L - 1) / 2) + 1 = ceil(L / 2)
```

After three convolutions:

```text
C(L) = ceil(L / 8)
```

A full 100-frame chunk therefore produces:

```text
100 -> 50 -> 25 -> 13 rows
```

The whole-audio post-CNN row count is consequently:

```text
M = 13 * floor(F / 100) + ceil((F % 100) / 8)
```

For integer arithmetic:

```text
M = 13 * (F // 100) + ((F % 100 + 7) // 8)
```

It is important not to replace this with `ceil(F / 8)`. The model performs
rounding independently inside every 100-frame chunk, so every full chunk
contributes 13 rows rather than 12.5 rows.

The installed vLLM formula is `_get_feat_extract_output_lengths()` in:

```text
.venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_asr.py
```

Our exact CPU mirror is
[`_expected_audio_output_lengths()`](../inference/vllm_qwen3_asr_optimizations/audio_cpu_metadata_pack_patch.py).

### Worked examples

| Decoded duration | Feature frames `F` | Calculation | Post-CNN rows `M` |
|---:|---:|---|---:|
| 1.0 s | 100 | `13 * 1 + 0` | 13 |
| 10.0 s | 1000 | `13 * 10 + 0` | 130 |
| 20.0 s | 2000 | `13 * 20 + 0` | 260 |
| 20.3 s | 2030 | `13 * 20 + ceil(30/8)` | 264 |
| 29.0 s | 2900 | `13 * 29 + 0` | 377 |
| 29.5 s | 2950 | `13 * 29 + ceil(50/8)` | 384 |
| 30.0 s | 3000 | `13 * 30 + 0` | 390 |

## Why the local-attention window is 104 rows

The number eight is a model configuration ratio, not an HTTP batch size and
not a new choice made by this optimization:

```text
n_window       = 50
raw chunk      = n_window * 2 = 100 feature frames
n_window_infer = 800 feature frames
chunks/window  = 800 // 100 = 8
```

Because each full raw chunk produces 13 post-CNN rows, one configured
inference attention window contains:

```text
window_aftercnn = 8 * 13 = 104 rows
```

The installed encoder computes the same value as:

```python
window_aftercnn = padded_mask_after_cnn.shape[-1] * (
    self.n_window_infer // (self.n_window * 2)
)
```

The transformer receives cumulative sequence boundaries for consecutive
windows. Examples are:

```text
M=264 -> lengths 104 + 104 + 56
         cu_seqlens = (0, 104, 208, 264)

M=390 -> lengths 104 + 104 + 104 + 78
         cu_seqlens = (0, 104, 208, 312, 390)
```

FlashAttention treats each interval as a separate local-attention sequence.
Our CPU sequence-metadata path reconstructs these boundaries on the CPU in
[`_build_cpu_metadata()`](../inference/vllm_qwen3_asr_optimizations/audio_cpu_metadata_pack_patch.py),
avoiding the previous GPU-to-CPU length readback while preserving the model's
existing algorithm.

## Which parts are general and which are shape-specialized

There are three distinct optimization scopes.

### CPU sequence metadata construction and Triton valid-row packing

The CPU sequence-metadata path derives raw chunk lengths, per-chunk post-CNN
lengths, packing offsets, and attention boundaries for arbitrary positive
feature lengths accepted by the guarded installed model implementation. This
part is not restricted to 20-second, 29-second, or 30-second audio.

The Triton valid-row pack copies only the valid CNN rows into their calculated
output offsets. Its CPU metadata tensor contains two halves: the valid row
count for each padded CNN chunk and that chunk's destination offset. One Triton
program is launched per padded row and masks invalid rows, eliminating the
prior boolean-index construction and GPU-to-host list materialization.

#### Why there are two CPU audio patches

The max-seqlen and sequence-metadata patches remove different synchronization
points at different seams:

| | CPU max-seqlen cache | CPU sequence metadata and row pack |
|---|---|---|
| Implementation | [`audio_cpu_maxseqlen_patch.py`](../inference/vllm_qwen3_asr_optimizations/audio_cpu_maxseqlen_patch.py) | [`audio_cpu_metadata_pack_patch.py`](../inference/vllm_qwen3_asr_optimizations/audio_cpu_metadata_pack_patch.py) |
| Patched seam | `compute_attn_mask_seqlen()` | Audio encoder `forward()` and multimodal field configuration |
| Produces | One CPU `int32` scalar with the guarded upper bound `104` | `chunk_lengths`, `pack_metadata`, `cu_seqlens`, and packed feature rows |
| Avoids | Repeated CUDA scalar readbacks in the 24 transformer layers | GPU/CPU length materialization, dynamic boolean indexing, and metadata synchronizations |
| Triton kernel | No | Yes, to compact valid post-CNN rows |
| Standalone | Can be enabled independently | Requires and installs the max-seqlen patch |

The max-seqlen patch is deliberately narrow. For the exact guarded encoder
geometry, it replaces `compute_attn_mask_seqlen(cu_seqlens)` with a cached
CPU tensor containing `104`. Downstream FlashAttention layers can therefore
read the upper bound without synchronizing on a CUDA scalar. It does not build
`cu_seqlens`, split audio, run convolutions, or move feature rows.

The sequence-metadata patch replaces the broader encoder input pipeline. It
keeps `audio_feature_lengths` on the CPU, constructs all chunking, packing, and
attention metadata there, runs the audio frontend, compacts valid rows
with Triton, transfers the small `cu_seqlens` tensor asynchronously, and then
runs the audio transformer. After constructing `cu_seqlens`, it still needs
a `max_seqlen` value for FlashAttention, which is why it depends on the
max-seqlen cache:

```text
CPU sequence-metadata patch
  -> build chunk_lengths, pack_metadata, and cu_seqlens
  -> Triton valid-row pack
  -> copy cu_seqlens to CUDA
  -> CPU max-seqlen cache returns 104
  -> run 24 audio-transformer layers
```

This dependency is enforced at installation. The broader patch requires both
`ASR_AUDIO_CPU_METADATA_PACK=1` and `ASR_AUDIO_CPU_MAXSEQLEN=1`, and it installs
the max-seqlen method replacement before replacing the encoder forward.

These are related but distinct pieces of state:

- `chunk_lengths` describes how the input feature frames are divided into raw
  convolution chunks.
- `pack_metadata` contains the valid post-CNN row count and destination offset
  for each padded chunk. The Triton kernel consumes this metadata to compact
  rows.
- `cu_seqlens` contains cumulative boundaries for FlashAttention. It partitions
  the already-packed rows into local-attention sequences; it is not a packed
  list of row lengths.
- The CPU max-seqlen cache supplies the largest interval between consecutive
  `cu_seqlens` values. For this encoder, `104` is the guarded safe upper bound.

For a canonical 30-second chunk, `F=3000` feature frames become 30 full
100-frame convolution chunks. Every chunk produces 13 valid post-CNN rows, so
the packed row count is `M=390`:

```text
chunk_lengths       = [100, 100, ..., 100]       # 30 entries
valid row counts    = [13, 13, ..., 13]          # 30 entries
destination offsets = [0, 13, 26, ..., 377]
packed output shape = [390, 1024]

cu_seqlens           = [0, 104, 208, 312, 390]
attention lengths    = [104, 104, 104, 78]
cached max_seqlen    = 104
```

The Triton pack uses the valid counts and destination offsets to create the
`[390, 1024]` tensor. FlashAttention then uses `cu_seqlens` to interpret that
same tensor as four independent local-attention sequences. No kernel “packs
sequence lengths”; the only packed objects are the valid post-CNN feature
rows.

Here, “valid” means that an output row corresponds to real feature frames
rather than right-padding added so chunks can be processed as one rectangular
tensor. It does not mean that the row contains speech or nonzero values.
For example, suppose one audio produces two raw chunks containing 100 and 30
feature frames. The shorter chunk is padded to 100 frames before the batched
convolutions:

```text
true input lengths             = [100, 30]
padded CNN input shape         = [2 chunks, 100 frames]
CNN output shape               = [2 chunks, 13 rows, 1024]
valid post-CNN rows per chunk  = [13, 4]
packing destination offsets    = [0, 13]
```

Three stride-two convolutions map the full chunk to 13 rows and the real
30-frame tail to `ceil(30 / 8) = 4` rows. The remaining nine rows in the
tail chunk exist only because that chunk was padded to 100 input frames. The
pack therefore copies all 13 rows from the first chunk followed by the first
four rows from the tail:

```text
packed output shape = [(13 + 4), 1024] = [17, 1024]
```

The CPU calculates `[13, 4]` and `[0, 13]` before the CNN runs because the
three convolutions' output-length formula is known. The Triton kernel applies
that metadata after the CNN, preserving the original chunk-major row order
while skipping only padding-derived rows.

The CPU sequence-metadata patch also retains `audio_feature_lengths` on the
CPU through vLLM's multimodal field configuration. It validates both
`feature_lens` and `aftercnn_lens`, constructs `chunk_lengths`,
`pack_metadata`, and `cu_seqlens`, and transfers only the small metadata needed
by GPU kernels. The CPU max-seqlen cache in
`audio_cpu_maxseqlen_patch.py` returns the already-known maximum attention
length, avoiding the same CUDA scalar readback in each of the 24 layers. The
cached value `104` is a safe upper bound; a very short request may have a
smaller actual attention partition.

Installation fails closed unless the installed distribution is vLLM
`0.24.0+cu129` and its `direct_url.json` wheel URL matches the wheel URL and
SHA-256 locked in `uv.lock`. A provenance mismatch prevents installation of
the metadata wrapper. After installation, runtime model/backend/tensor/length
guards protect the narrower forward seam; a runtime miss moves the CPU length
tensors back to the feature device and delegates that invocation to the
original vLLM forward.

Implementation:

- [`audio_cpu_metadata_pack_patch.py`](../inference/vllm_qwen3_asr_optimizations/audio_cpu_metadata_pack_patch.py)
- [`audio_cpu_maxseqlen_patch.py`](../inference/vllm_qwen3_asr_optimizations/audio_cpu_maxseqlen_patch.py)

### Audio frontend CUDA graphs

The frontend graph covers convolution, projection, positional embedding, and
valid-row packing. It validates the complete runtime key before replay:

- padded shape and stride;
- dtype and device;
- every raw chunk length;
- every pack length and output offset;
- the full `feature_lens` and `aftercnn_lens` tuples;
- all cumulative attention boundaries.

The natural family admits exactly one audio with `F=2897..3000`. Those 104
exact feature-length keys reduce to 14 graph-static signatures, one for each
post-CNN row count `M=377..390`. The signatures share a CUDA graph memory pool
and are serialized because captures from that pool can alias allocations.

All other shapes remain eager. In particular, the 21-chunk, 20--21 second tail
family from the `opt6` experiment is no longer admitted, even if its metadata
is otherwise compatible.

Implementation:

- [`audio_frontend_cudagraph_patch.py`](../inference/vllm_qwen3_asr_optimizations/audio_frontend_cudagraph_patch.py)
- [Detailed frontend design](audio-frontend-cudagraph.md)

### Audio transformer CUDA graphs

The transformer graph covers the 24 audio-transformer layers and output projection.
Exact runtime keys are admitted into one padded graph family:

```text
natural rows M in {377, ..., 390}
  -> padded graph bucket of 390 rows
```

Padding reduces 14 exact transformer shapes to one captured graph, while exact
pre-admission prevents an unsupported shape or attention layout from replaying
through a merely equal-sized bucket.

Implementation:

- [`audio_transformer_cudagraph_patch.py`](../inference/vllm_qwen3_asr_optimizations/audio_transformer_cudagraph_patch.py)
- [Detailed transformer design](../ideas/audio_transformer_cudagraph.md)

## Exact natural graph coverage

Natural graph admission covers the following continuous feature-length range:

```text
2897 <= F <= 3000
```

This is greater than 28.96 through 30.00 seconds for chunks supplied by the
server API, whose maximum chunk duration is 30 seconds.

| Post-CNN rows `M` | Exact feature lengths `F` | Approximate duration |
|---:|---:|---:|
| 377 | 2897..2900 | `(28.96, 29.00]` s |
| 378 | 2901..2908 | `(29.00, 29.08]` s |
| 379 | 2909..2916 | `(29.08, 29.16]` s |
| 380 | 2917..2924 | `(29.16, 29.24]` s |
| 381 | 2925..2932 | `(29.24, 29.32]` s |
| 382 | 2933..2940 | `(29.32, 29.40]` s |
| 383 | 2941..2948 | `(29.40, 29.48]` s |
| 384 | 2949..2956 | `(29.48, 29.56]` s |
| 385 | 2957..2964 | `(29.56, 29.64]` s |
| 386 | 2965..2972 | `(29.64, 29.72]` s |
| 387 | 2973..2980 | `(29.72, 29.80]` s |
| 388 | 2981..2988 | `(29.80, 29.88]` s |
| 389 | 2989..2996 | `(29.88, 29.96]` s |
| 390 | 2997..3000 | `(29.96, 30.00]` s |

The full natural interval is `(28.96, 30.00]` seconds after decoding to 16 kHz.

## Historical tail experiment and current exclusion

The 50-second workload produced a second family of approximately 20--21 second
final chunks. Follow-up branch `opt6/audio-tail-rows-263-271` made the admitted
single-audio feature range continuous across these rows:

| Post-CNN rows `M` | Exact feature lengths `F` | Approximate duration |
|---:|---:|---:|
| 263 | 2017..2024 | `(20.16, 20.24]` s |
| 264 | 2025..2032 | `(20.24, 20.32]` s |
| 265 | 2033..2040 | `(20.32, 20.40]` s |
| 266 | 2041..2048 | `(20.40, 20.48]` s |
| 267 | 2049..2056 | `(20.48, 20.56]` s |
| 268 | 2057..2064 | `(20.56, 20.64]` s |
| 269 | 2065..2072 | `(20.64, 20.72]` s |
| 270 | 2073..2080 | `(20.72, 20.80]` s |
| 271 | 2081..2088 | `(20.80, 20.88]` s |
| 272 | 2089..2096 | `(20.88, 20.96]` s |
| 273 | 2097..2100 | `(20.96, 21.00]` s |

That experimental family covered `F=2017..2100`, or `(20.16, 21.00]` seconds,
without changing the 273-row transformer bucket. The
promoted natural-only implementation sets the tail admission sets to empty,
so every row in this table now takes the general eager path. The mapping is
retained here to explain the experiment and the fixed-50 benchmark stress case.

### Historical `opt6` GPU validation

On 2026-07-21, GPU1 passed the SM90 transformer helper for all 25 exact keys:
the
11 tail rows `M=263..273` and 14 natural rows `M=377..390`. Every key was
bitwise exact, bucket-eager and replay kernel order matched at 268 kernels,
and alternating plus two-thread/two-stream shared-bucket gates passed.

The chained audio-encoder helper then passed each newly admitted row with
eight-observation probation, changed-content equality, and two-thread/two-
stream concurrency:

| New row | Eager CUDA | Copy + graph replay + clone | Eager host call | Graph host call |
|---:|---:|---:|---:|---:|
| 263 | 6493.632 us | 2134.784 us | 6507.907 us | 165.693 us |
| 266 | 6804.240 us | 2135.920 us | 6834.688 us | 170.253 us |
| 269 | 6683.440 us | 2137.824 us | 6722.245 us | 172.673 us |
| 271 | 6722.048 us | 2133.536 us | 6700.560 us | 167.908 us |

These focused helper measurements were followed by a clean end-to-end service
run from commit `98f6992` on GPU1. The 500-file, 50-second batched workload
completed with zero failures/timeouts at 29.572 files/s, 0.526 s mean latency,
and 0.199 s mean TTFT. The server log also showed the expanded 21-chunk family
entering probation and replay through the existing 273-row transformer bucket.

## Long-file splitting and arbitrary final tails

The vLLM transcription server first decodes and resamples the complete file.
When its duration is greater than 30 seconds, it repeatedly searches for a
low-energy split point during the final second of the next 30-second region.

With the current configuration:

```text
sample rate               = 16000 Hz
maximum chunk             = 30 seconds
split search region       = [29, 30) seconds relative to the chunk start
energy window             = 1600 samples = 0.1 seconds
candidate non-final sizes = 29.0, 29.1, ..., 29.8 seconds
```

Despite the configuration name `overlap_chunk_second`, the current splitter
uses the one-second interval as a split-search region; it does not duplicate
that audio in both emitted chunks. The next chunk begins at the selected split
point.

Every non-final chunk therefore falls inside the natural graph range. Once the
remaining audio is at most 30 seconds, the server emits it unchanged as the
final chunk. For an arbitrary long file:

```text
0 < final tail duration <= 30 seconds
```

Consequently, full CUDA-graph coverage cannot be inferred from total file
duration alone. Audio energy affects the chosen split position, and the final
tail can have any positive duration up to 30 seconds. Unsupported tails remain
correct and run through the eager fallback; earlier eligible chunks from the
same file can still use their graphs.

### Exact 50-second example

For an exact 50-second decoded input with one split:

| First chunk | Final tail | Tail rows | Current tail graph? |
|---:|---:|---:|:---:|
| 29.0 s | 21.0 s | 273 | No |
| 29.1 s | 20.9 s | 272 | No |
| 29.2 s | 20.8 s | 270 | No |
| 29.3 s | 20.7 s | 269 | No |
| 29.4 s | 20.6 s | 268 | No |
| 29.5 s | 20.5 s | 267 | No |
| 29.6 s | 20.4 s | 265 | No |
| 29.7 s | 20.3 s | 264 | No |
| 29.8 s | 20.2 s | 263 | No |

The non-final chunk is graph-eligible in every row. The final tail is now
always eager; the previous branches graphed seven and then all nine of these
fixed-50 tail shapes.

### Other illustrative file lengths

| Original duration | Typical server chunks | Likely graph behavior |
|---:|---|---|
| 35 s | approximately 29.x + 5.x s | natural graph, then eager tail |
| 40 s | approximately 29.x + 10.x s | natural graph, then eager tail |
| 50 s | approximately 29.x + 20.x s | natural graph, then eager tail |
| 59 s | approximately 29.x + 29.x s | both chunks usually natural graphs |
| 65 s | approximately 29.x + 29.x + 6.x s | two natural graphs, then eager tail |

These are illustrations rather than admission guarantees because the actual
low-energy split depends on the waveform.

## Why the varying 550-file workload still improves

The full 550-file natural workload is not restricted to one audio duration.
The refreshed 16-worker row selected by `analyse_results.py` is:

| Path | Throughput | Latency p50/p95/p99 | TTFT p50/p95/p99 | CER/WER |
|---|---:|---:|---:|---:|
| FP8 static + Q/K fusion | 4.604 files/s | 3.137 / 6.276 / 8.382 s | 0.385 / 1.069 / 1.761 s | 0.166 / 0.387 |
| Final natural-only graphs | 5.722 files/s | 2.521 / 4.830 / 6.653 s | 0.543 / 1.328 / 1.759 s | 0.160 / 0.382 |
| Change | +24.28% | -19.64% / -23.04% / -20.63% | +41.04% / +24.23% / -0.11% | snapshot comparison |

This result demonstrates useful partial coverage, not uniform acceleration of
every file:

- non-final 29--30 second chunks from long recordings frequently enter the
  natural frontend and transformer graphs;
- eligible final chunks also enter a graph;
- unsupported final tails retain the general CPU metadata path and use eager
  frontend/transformer execution;
- graph-eligible invocations occur often enough to improve aggregate batched
  throughput and mean latency;
- TTFT p50/p95 regress in this comparison and remain the main measured
  tradeoff; TTFT p99 is effectively unchanged.

The workload result therefore supports the description **shape-specialized
but workload-general**. The implementation is not tied to the 550 source files
or their contents, but graph replay is restricted to the admitted runtime
shapes.

Quality measurements, all four analyzer tables, warm-up boundaries, and the
historical-versus-final distinction are recorded in
[the final natural-only benchmark note](audio-natural-only-cudagraph-benchmark.md).

## Cold admission versus steady replay

Graph eligibility does not mean that the first occurrence immediately replays
a graph. Each exact runtime key passes probation:

```text
observations 1..7 -> eager execution
observation 8     -> capture or alias validation against eager output
observation 9+    -> hot graph replay when admission succeeded
```

Capture errors, numerical mismatches, changed metadata, unsupported layouts,
training or gradient mode, and cache-capacity limits all fail closed to eager
execution. This is why same-server steady measurements best represent the hot
graph path, while fresh-service measurements include admission overhead.

## Multi-audio and batch-layout qualification

HTTP concurrency does not weaken graph validation. An unseen single audio with
an admitted length and canonical metadata can use both graphs regardless of
its content.

Frontend admission explicitly requires one audio and checks `feature_lens`,
`aftercnn_lens`, raw chunking, pack offsets, and `cu_seqlens`; a multi-audio
frontend therefore stays eager even when its total rows equal an admitted `M`.
The transformer sees only the boundary tensor and attention metadata. It may qualify
if those values exactly equal `[M,1024]` plus canonical
`(0,104,208,312,M)`, but equal total rows with a different partition remain a
miss. In either case, a miss retains eager correctness.

## Small calculator

The following reproduces the single-chunk length mapping after decoding to
16 kHz:

```python
def qwen3_asr_lengths(samples_16khz: int) -> tuple[int, int]:
    feature_frames = (samples_16khz + 159) // 160
    full_chunks, tail_frames = divmod(feature_frames, 100)
    post_cnn_rows = 13 * full_chunks + (tail_frames + 7) // 8
    return feature_frames, post_cnn_rows
```

For runtime admission, these two integers are necessary but not sufficient.
The implementation additionally validates tensor layout, every chunk and pack
offset, the per-audio length tuples, and the cumulative attention boundaries.

## Practical conclusions

- Raw audio content does not select the fast path; decoded shape and metadata
  do.
- The processor right-pads to the next 10 ms hop, so feature length is the
  ceiling of decoded 16 kHz samples divided by 160.
- The three CNNs produce 13 rows per full one-second feature chunk.
- Eight such chunks form the configured 104-row local-attention window.
- Non-final chunks created by the current long-file splitter naturally land in
  the 29--30 second graph family.
- The final chunk of a long file can be anywhere in `(0, 30]` seconds; the
  promoted path intentionally leaves every non-natural final tail eager.
- Unsupported shapes use safe eager fallback rather than an approximate graph.
- The varying 550-file workload benefits because enough of its individual
  server chunks are graph-eligible, even though not every audio is fully
  covered.
