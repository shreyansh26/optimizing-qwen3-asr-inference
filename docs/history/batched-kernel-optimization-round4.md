# Batched kernel optimization round 4

Audience: developers working on Qwen3-ASR/vLLM batched inference kernels.

> **Historical experiment record.** This note preserves the candidate-selection
> evidence from round four, including the earlier 20--21 second tail graph and
> two-bucket suffix design. The promoted branch now uses 104 natural prefix
> keys, 14 prefix signatures, and one 390-row transformer graph; all tail shapes are
> eager. Current architecture and measurements are in the
> [audio-length guide](../qwen3-asr-audio-length-and-graph-fast-path.md) and
> [final benchmark report](../audio-natural-only-cudagraph-benchmark.md).

## Decision at the end of round four

Promote `opt5/audio-prefix-shared-suffix-bucketed` as the cleanest latency-first
round-four candidate tested so far.

This branch combines:

- the accepted static-FP8 + Q/K RMSNorm + MRoPE + KV-cache fused decoder path;
- CPU audio max-seqlen and CPU metadata valid-row packing;
- a shared memory pool for exact natural audio-prefix CUDA graphs;
- a two-bucket exact audio-suffix CUDA graph cache.

No PyTorch, vLLM, Triton, CUDA, or model version was changed.

For the complete sample-to-feature-to-CNN length derivation, the origin of the
104-row attention window, exact graph-eligible duration ranges, and long-file
tail behavior, see
[Qwen3-ASR audio lengths and CUDA-graph fast-path coverage](../qwen3-asr-audio-length-and-graph-fast-path.md).

## Clean service measurements

Run all service tests on one explicitly free GPU. The benchmark runner resets
the vLLM prefix cache itself before warmup and measured inference. Do not rely
on unconfirmed server reset endpoints for multimodal or encoder caches.

| Path | Workload | Throughput | Avg latency | Avg TTFT | Notes |
|---|---:|---:|---:|---:|---|
| Accepted adjacent repeat 1 | 550 natural | 4.766 files/s | 3.292 s | 0.551 s | `opt3/audio-cpu-metadata-pack` |
| Accepted adjacent repeat 2 | 550 natural | 4.839 files/s | 3.245 s | 0.515 s | `opt3/audio-cpu-metadata-pack` |
| Accepted adjacent mean | 550 natural | 4.803 files/s | 3.269 s | 0.533 s | comparison baseline |
| Bucketed suffix only | 550 natural | 5.651 files/s | 2.772 s | 0.667 s | clean suffix-isolation service run |
| Shared-prefix + bucketed suffix | 550 natural, fresh | 5.576 files/s | 2.810 s | 0.666 s | clean integration run |
| Shared-prefix + bucketed suffix | 550 natural, same-server steady | 5.809 files/s | 2.696 s | 0.629 s | clean integration run; selected winner |
| Bucketed suffix only | 500 fixed-50 | 30.012 files/s | 0.506 s | 0.158 s | clean fixed-length service run |

Selected comparison against the accepted adjacent mean:

```text
throughput: 4.803 -> 5.809 files/s  (+20.95%)
latency:    3.269 -> 2.696 s        (-17.53%)
TTFT:       0.533 -> 0.629 s        (+18.01%)
```

The suffix-only fixed-50 result also beat the accepted fixed-50 control
`25.345 files/s, 0.615 s latency, 0.184 s TTFT` by:

```text
throughput: +18.41%
latency:    -17.72%
TTFT:       -14.13%
```

## Quality status

Full 550-file manual scoring shows modest degradation versus the adjacent
accepted repeat, while the exact helper gates remain bitwise.

| Path | CER | WER | Notes |
|---|---:|---:|---|
| Historical accepted full run | 0.163614 | 0.384638 | older `opt3/audio-cpu-metadata-pack` reference |
| Accepted adjacent repeat | 0.164445 | 0.385197 | adjacent control for this round |
| Shared-prefix + bucketed suffix | 0.165517 | 0.385297 | selected round-four candidate |
| Prefix-bucket candidate | 0.163100 | 0.384623 | service run contaminated; do not promote from this score alone |

The selected candidate changed quality by:

```text
vs accepted adjacent repeat: +0.001072 CER, +0.000100 WER
vs historical accepted:      +0.001903 CER, +0.000659 WER
```

Treat this as acceptable only for the latency-prioritized batched track. If the
deployment gate is stricter than this, rerun canonical full scoring with more
adjacent repeats before merging.

## Bucketed suffix CUDA graph

Branch: `opt4/audio-suffix-cudagraph-bucketed`

Commit: `d2d0cf78e5ff98d0ca84ba50fbbe5834ddfc6091`

The suffix cache maps 21 canonical suffix runtime keys into two graph buckets:

```text
tail M in {264, 265, 267, 268, 270, 272, 273}
  graph cu_seqlens = (0, 104, 208, M, 273)

natural M in {377, ..., 390}
  graph cu_seqlens = (0, 104, 208, 312, M, 390)
```

The helper passed on GPU1:

```text
gate=PASS_BUCKETED_AUDIO_TRANSFORMER_CUDAGRAPH
```

Evidence:

- all 21 exact suffix keys passed bitwise admission;
- both duplicate terminal endpoint cases passed:
  `(0,104,208,273,273)` and `(0,104,208,312,390,390)`;
- eager and captured paths had identical 268-kernel order;
- alternating cross-key retained-output replay passed;
- concurrent two-thread/two-stream replay passed;
- tail replay plus input/metadata copy was about 1.507-1.525 ms versus eager
  about 7.21-7.31 ms;
- natural replay plus input/metadata copy was about 1.641-1.663 ms versus eager
  about 7.16-7.27 ms.

## Audio frontend CUDA graph candidates

### Shared-prefix pool

Branch: `opt4/audio-prefix-cudagraph-natural-shared`

Commit: `4a2896a1fced4923ebd1867fa3f38c6efabf0a9f`

This keeps the 14 natural prefix signatures but shares one CUDA graph memory
pool and transaction lock across the cache.

The helper passed on GPU1:

```text
gate=PASS_EXACT_NATURAL_AUDIO_FRONTEND_CUDAGRAPH
```

Memory after the 14-signature helper gate dropped from the original natural
prefix implementation's 5,906 MiB reserved pool footprint to 1,150 MiB
reserved. Allocated graph memory after all 14 signatures was 118.964 MiB.

This is the prefix implementation used in the selected
`opt5/audio-prefix-shared-suffix-bucketed` service runs.

### Prefix bucket

Branch: `opt4/audio-prefix-cudagraph-bucketed`

Status: GPU helper passed, but clean service validation is still pending because
GPU1 was reclaimed by unrelated users before the follow-up service run could be
completed.

The prefix bucket maps 104 exact natural keys to one fixed
`(30,1,128,100) -> 390-row` graph.

Helper evidence:

- 100 exact keys admitted;
- the four `M=377` keys failed closed, as expected from the 29-chunk versus
  30-chunk algorithm ambiguity;
- one graph signature was retained;
- helper memory was 102.666 MiB allocated and 1,130 MiB reserved.

The observed prefix-bucket quality score was `0.163100 CER / 0.384623 WER`,
but the matching service timing run was contaminated by unrelated GPU activity.
Do not use that timing result for selection.

## Combined stacks tested

### Shared-prefix + bucketed suffix

Branch: `opt5/audio-prefix-shared-suffix-bucketed`

Commits:

- `24f2ee8`: bucketed suffix integration
- `9dde02e`: shared-prefix pool integration

Clean service result:

```text
fresh 550 natural:       5.576 files/s, 2.810 s latency, 0.666 s TTFT
same-server steady 550:  5.809 files/s, 2.696 s latency, 0.629 s TTFT
```

This is the current round-four winner because it produced the best clean
latency and throughput result without the graph-pool OOM seen in the earlier
naive prefix+suffix integration.

### Prefix-bucket + suffix-bucket

Branch: `opt5/audio-prefix-bucketed-suffix-bucketed`

Status: helper pieces pass individually; clean service validation is pending
because GPU1 became unavailable.

### Fast-admit prefix-bucket stack

Branch: `opt6/audio-prefix-suffix-bucketed-fast-admit`

Commit: `2fb8395085827643c1dc8467f6a9e61cf7392ecc`

Status: GPU/service validation is pending because GPU1 became unavailable.
This branch should not be compared against the selected winner until it has the
same adjacent accepted/candidate service controls.

## Rejected or not promoted

| Candidate | Decision |
|---|---|
| all-convolution bias+GELU fusion | Exact/microbench favorable, but service regressed. |
| projection layout-only tweak | Helper gain was about 0.226%; not enough to promote. |
| clustered split-K decoder down projection | Runtime failed with illegal instruction 715. |
| audio channels-last path | Neutral or worse. |
| CUTLASS batch-invariant tile | 29-35% slower. |
| Triton attention static-output path | Service latency regressed by about 3.14%. |
| FP8 LM head | Rejected. |
| producer RMS/static quant fusion | Already handled by Inductor fusion in the active path. |
| CuTe gate/up + SwiGLU + FP8 path | Runtime failed with illegal instruction 715; fail-closed. |
| built-in vLLM encoder graph/compile knobs | Not configuration-viable for this path. |
| residual + LayerNorm exploration | Not accepted; one service attempt hit graph-pool memory pressure. |

## Commands

Use the shared project virtual environment for every command:

```bash
export UV_PROJECT_ENVIRONMENT=/mnt/ssd1/shreyansh/home_dir/asr_experiments/.venv
export MODEL=/mnt/ssd2/hf_models/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5
```

### Launch the selected candidate

```bash
cd /mnt/ssd1/shreyansh/home_dir/asr_experiments_worktrees/audio-prefix-shared-suffix-bucketed
CUDA_VISIBLE_DEVICES=1 \
PORT=8091 \
MODEL="$MODEL" \
UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT" \
bash inference/run_vllm_fp8_static_audio_encoder_cudagraphs.sh \
  --gpu-memory-utilization 0.75
```

> **Note:** Do not use a worktree-local `.venv`. If another server is running on
> the host, also set an isolated `HF_HOME` before launch.

### Run a 550-file natural performance pass

```bash
cd /mnt/ssd1/shreyansh/home_dir/asr_experiments_worktrees/audio-prefix-shared-suffix-bucketed
UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT" \
uv run inference/run_benchmark.py \
  --mode batched \
  --workers 16 \
  --overwrite \
  --num-files 550 \
  --input-root /mnt/ssd1/shreyansh/home_dir/asr_experiments/data/prepared_data \
  --output-root /tmp/asr_round4_shared_prefix_bucketed_suffix_550 \
  --base-url http://127.0.0.1:8091/v1
```

### Run fixed-50

```bash
cd /mnt/ssd1/shreyansh/home_dir/asr_experiments_worktrees/audio-prefix-shared-suffix-bucketed
UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT" \
uv run inference/run_benchmark.py \
  --mode batched \
  --workers 16 \
  --overwrite \
  --uniform-audio-length 50 \
  --num-files 500 \
  --input-root /mnt/ssd1/shreyansh/home_dir/asr_experiments/data/prepared_data \
  --output-root /tmp/asr_round4_shared_prefix_bucketed_suffix_fixed50 \
  --base-url http://127.0.0.1:8091/v1
```

### Run canonical full quality

```bash
cd /mnt/ssd1/shreyansh/home_dir/asr_experiments_worktrees/audio-prefix-shared-suffix-bucketed
UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT" \
uv run inference/run_benchmark.py \
  --mode batched \
  --workers 16 \
  --overwrite \
  --input-root /mnt/ssd1/shreyansh/home_dir/asr_experiments/data/prepared_data \
  --output-root /tmp/asr_round4_shared_prefix_bucketed_suffix_full_quality \
  --base-url http://127.0.0.1:8091/v1
```

## Promotion gate

Promote a later branch over `opt5/audio-prefix-shared-suffix-bucketed` only if
all conditions hold:

1. The exact helper gates pass on the final branch.
2. The service log shows expected captures/admissions before measurement and no
   fail-open or OOM.
3. No unrelated GPU jobs overlap the measured service window.
4. Adjacent accepted/candidate controls confirm a latency win.
5. TTFT and throughput are reported even when latency is the priority metric.
6. Canonical no-`--num-files` CER/WER does not materially regress.
