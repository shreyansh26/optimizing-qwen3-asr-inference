# Bucketed audio suffix CUDA graph experiment

## Status

> **Historical design note.** The two-bucket, 25-key tail experiment below was
> validated but intentionally not promoted. The final path admits only natural
> `M=377..390` layouts and maps all 14 exact keys to one padded 390-row graph;
> every 20--21 second tail stays eager. See the
> [final implementation guide](../docs/qwen3-asr-audio-length-and-graph-fast-path.md)
> and [benchmark report](../docs/audio-natural-only-cudagraph-benchmark.md).

The original candidate was developed on branch
`opt4/audio-suffix-cudagraph-bucketed` and later integrated into selected
round-four branch `opt5/audio-prefix-shared-suffix-bucketed`.

Follow-up branch `opt6/audio-tail-rows-263-271` expanded the tail family to
`M=263..273`. On 2026-07-21, GPU1 passed all 25 exact suffix keys with bitwise
equality, identical 268-kernel bucket order, alternating retained-output
checks, and two-thread/two-stream concurrency. The family was later excluded
from promotion to avoid specializing the persistent cache inventory to the
fixed-50 benchmark's derived final tails.

No PyTorch, vLLM, Triton, or other dependency version changed.

## Eager-path audit

`run_audio_suffix_eager` has no row-dependent Python control flow. It executes:

```text
24 encoder layers
-> ln_post
-> proj1
-> act
-> proj2
-> [M, 2048]
```

Every encoder layer applies per-row normalization/MLP work and one packed
variable-length self-attention call. Therefore padding rows can be isolated
from the real rows by appending them as a separate attention sequence. The
original unpadded eager callable remains the fallback and admission reference.

The installed vLLM also documents and uses repeated terminal cumulative
offsets for CUDA-graph vision metadata: right-padding `cu_seqlens` with its
last value represents zero-length sequences ignored by varlen attention. That
supports the proposed `M == bucket` topology, but the real Qwen3-ASR/FA3 path
must still pass the GPU bitwise gate before this experiment is usable.

## Exact requests and two graph buckets

The follow-up `opt6/audio-tail-rows-263-271` branch considers 25 canonical
runtime keys:

```text
tail family (11 exact keys):
  M in {263, ..., 273}
  runtime cu_seqlens == (0, 104, 208, M)

natural family (14 exact keys):
  M in {377, ..., 390}
  runtime cu_seqlens == (0, 104, 208, 312, M)

both:
  max_seqlen == 104
  contiguous CUDA BF16 hidden_states[M, 1024]
```

They map to at most two persistent graphs:

```text
tail graph:
  bucket rows = 273
  graph cu_seqlens = (0, 104, 208, M, 273)

natural graph:
  bucket rows = 390
  graph cu_seqlens = (0, 104, 208, 312, M, 390)
```

For `M < bucket`, `[M, bucket)` is one independent padding sequence. For the
maximum key, the final two offsets are equal:

```text
tail M=273:    (0, 104, 208, 273, 273)
natural M=390: (0, 104, 208, 312, 390, 390)
```

The graph topology and metadata tensor length are fixed within each family.
Only the runtime prefix values change. The padding segment length is at most 10
rows for the tail graph and 13 rows for the natural graph, so `max_seqlen=104`
remains a valid upper bound.

The workload audit behind the natural family found 5,698 of 6,161 calls
(92.5%) with `cu_seqlens.numel() == 5`; its top nine row buckets within
377-387 cover 91.2%. These are workload counts, not performance evidence.

## Stable buffers, ownership, and concurrency

Each bucket entry retains:

```text
static_hidden_states[bucket, 1024]
static_cu_seqlens[runtime_numel + 1]
static_max_seqlen scalar
captured output[bucket, 2048]
CUDAGraph
dedicated execution stream
replay lock
```

Capture initializes padding rows to zero. A hot replay copies only the first
`M` hidden rows and the runtime `cu_seqlens` prefix. The final bucket endpoint
is initialized once and remains stable. Padding input rows can retain values
written by a previous larger exact key because they remain in a separate
attention sequence and every other suffix operation is per-row; they cannot
affect the returned real prefix. The per-key bitwise gate verifies this
assumption on the real module stack before admission.

Every replay clones only `output[:M]`, so callers own independent `[M, 2048]`
storage. The caller hidden tensor and `cu_seqlens` tensor are both recorded on
the entry stream until their asynchronous copies finish.

All exact keys in one family share mutable graph buffers. A per-entry lock
therefore serializes the complete transaction:

```text
copy hidden prefix
-> copy runtime cu_seqlens prefix
-> graph replay
-> clone output[:M]
```

The dedicated entry stream supplies FIFO GPU ordering across callers. Caller
to entry and entry to caller stream waits preserve producer/consumer ordering
without a hot-path device synchronization. Cross-key calls sharing a bucket
use the same lock; the two different bucket entries may operate independently.
The cache lock serializes cold capture/admission, and the global attachment lock
still prevents two caches from being attached to one encoder concurrently.

## Exact probation and fail-closed admission

Probation remains per exact runtime key, not per bucket:

```text
observations 1-7 -> original unpadded eager
observation 8:
  capture the family bucket if it does not exist
  replay the bucket for this exact M
  compare output[:M] with original unpadded eager using torch.equal
observation 9+ -> replay only if that exact M passed
```

The first admitted key in a family creates the graph. Every later M reuses it
but must independently reach observation eight and pass the bitwise comparison.
A mismatch rejects only that exact key permanently; it continues on unpadded
eager. A structurally valid bucket graph may remain for other exact keys, which
must pass their own gates. Capture exceptions, unsupported layouts, wrong
topology, training/grad mode, nested capture, wrong layer count, capacity
overflow, or output shape/dtype/device drift all fail closed.

The observation-eight call returns the independently allocated eager result.
No graph-owned output is returned directly. `torch.equal` synchronizes only
during cold exact-key admission; admitted hot replays do not compare.

Expected logs distinguish bucket capture from exact admission:

```text
Bucketed audio suffix CUDA graph probation uses eager suffix cu_seqlens=<runtime> bucket_rows=<273|390> observation=<n>/8 bucket_occupancy=<n>/2 admitted=<n>/25
Captured bucketed audio suffix CUDA graph graph_cu_seqlens=<graph tuple> bucket_rows=<273|390> observation=8/8 bucket_occupancy=<n>/2 capture_duration_ms=<ms> cumulative_replays=0
Admitted bitwise-exact bucketed audio suffix cu_seqlens=<runtime> graph_cu_seqlens=<graph tuple> bucket_rows=<273|390> observation=8/8 admitted=<n>/25
ASR bucketed audio suffix CUDA graph replay active cu_seqlens=<runtime> bucket_rows=<273|390> observation=8/8 bucket_cumulative_replays=<n>
```

## Environment and launcher

The strict gate remains:

```text
ASR_AUDIO_SUFFIX_CUDAGRAPH=1
```

It requires:

```text
ASR_AUDIO_CPU_MAXSEQLEN=1
ASR_AUDIO_CPU_METADATA_PACK=1
```

Launcher:

```bash
inference/experimental_launchers/run_vllm_fp8_static_qk_prefill_audio_suffix_cudagraph.sh
```

## CUDA helper gate

Do not run without an explicitly allocated SM90 GPU:

```bash
cd /mnt/ssd1/shreyansh/home_dir/asr_experiments_worktrees/audio-suffix-cudagraph-bucketed
rtk env \
  CUDA_VISIBLE_DEVICES=<free-gpu> \
  UV_PROJECT_ENVIRONMENT=/mnt/ssd1/shreyansh/home_dir/asr_experiments/.venv \
  uv run inference/vllm_static_fp8/benchmarks/bench_audio_suffix_cudagraph.py \
    --warmup 3 \
    --repeats 10 \
    --concurrency-iterations 20 \
    --master-port <free-port>
```

The default helper covers all 25 exact keys. It requires:

- observations one through seven to remain eager for every exact key;
- every exact M to pass unpadded-eager versus bucket-replay bitwise equality;
- exactly two captures and 25 exact admissions;
- both maximum-row duplicate-endpoint cases to pass;
- bucket-eager versus captured-graph CUDA kernel ordering to match;
- replay timings to include hidden-prefix copy, metadata-prefix copy, graph
  launch, and independent prefix clone;
- alternating retained outputs from two exact keys sharing one bucket to remain
  bitwise exact;
- two different exact keys sharing one bucket to race from two host threads and
  two CUDA streams without corruption.

Required terminal lines:

```text
alternating_cross_key_gate=PASS keys=2 requests=4 ...
concurrent_cross_key_gate=PASS keys=2 threads=2 streams=2 ...
gate=PASS_BUCKETED_AUDIO_SUFFIX_CUDAGRAPH
```

No service benchmark is valid unless all 25 admissions complete before the
measured interval and no capture, rejection, or capacity warning appears during
measurement.

## Promotion gate

1. Pass the all-25 CUDA helper, including duplicate endpoints, exact admission,
   alternating ownership, and cross-key concurrency.
2. Run adjacent accepted/candidate/accepted B16 controls on one GPU and confirm
   two bucket captures plus all observed exact admissions before measurement.
3. Reject if priority average latency is neutral or worse, even if TTFT or host
   enqueue time improves.
4. Only after a latency win, run full 550-file batched CER/WER and compare
   throughput, latency, TTFT, CER, and WER with the accepted path.
