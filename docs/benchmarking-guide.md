# Benchmarking and CUDA helper guide

This is the command reference for the repository's benchmarking entry points.
It separates three kinds of evidence that answer different questions:

1. **End-to-end service benchmarks** measure latency, TTFT, throughput, and
   optionally CER/WER against a running vLLM server.
2. **CUDA helper benchmarks** use synthetic tensors to prove that an isolated
   optimization is correct and to measure its host and GPU costs.
3. **Nsight Systems captures** record the kernel and CUDA-graph execution of a
   running server for low-level investigation.

A CUDA helper passing is not a service-level performance or quality result.
Conversely, an end-to-end improvement should not be promoted until the
corresponding exactness helper also passes.

## Entry-point map

| File | Scope | Requires server? | Uses audio files? | Primary result |
|---|---|:---:|:---:|---|
| `inference/run_benchmark.py` | End-to-end sequential or concurrent inference | Yes | Yes | Latency, TTFT, throughput, CER/WER |
| `inference/analyse_results.py` | Aggregate recorded service rows | No | No | Comparison tables |
| `bench_audio_cpu_maxseqlen.py` | CPU max-seqlen substitution | No | No | Attention correctness and avoided scalar-readback cost |
| `bench_audio_cpu_metadata_pack.py` | CPU metadata and Triton valid-row pack | No | No | Bitwise packing, synchronization events, timing |
| `bench_audio_frontend_cudagraph.py` | Audio frontend CUDA graph cache | No | No | Exactness, kernel order, timing, ownership, concurrency |
| `bench_audio_transformer_cudagraph.py` | Audio transformer CUDA graph cache | No | No | Exactness, kernel order, timing, bucket reuse, concurrency |
| `bench_audio_encoder_cudagraphs.py` | Chained frontend and transformer caches | No | No | End-to-end encoder graph exactness and timing |
| `start_vllm_server_with_nsys.sh` | Launch server under Nsight | Starts it | Indirectly | `.nsys-rep` and SQLite trace |
| `run_nsys_profile.sh` | Warm up and trigger Nsight capture | Yes | Yes | One bounded profiled request |

The five `bench_audio_*.py` helpers live under
`inference/vllm_qwen3_asr_optimizations/benchmarks/`.

## Common setup

Run commands from the repository root and use the shared project environment:

```bash
cd /mnt/ssd1/shreyansh/home_dir/asr_experiments

export UV_PROJECT_ENVIRONMENT=/mnt/ssd1/shreyansh/home_dir/asr_experiments/.venv
```

For CUDA helpers, select one explicitly free GPU:

```bash
export CUDA_VISIBLE_DEVICES=<free-gpu>
```

The graph helpers set their internal device to `cuda:0`. With
`CUDA_VISIBLE_DEVICES=<free-gpu>`, that is the selected physical GPU. Do not
run a vLLM server or another memory-intensive process on the same GPU.

The frontend, transformer, and combined graph gates require SM90, normally an
NVIDIA H100. The transformer and combined helpers also initialize a one-process NCCL group,
so their `--master-port` must be unused.

## End-to-end service benchmark

### `run_benchmark.py`

This wrapper selects `run_infer.py` for sequential mode or
`run_infer_batched.py` for batched mode. It streams the child process output,
extracts request counts and performance metrics, optionally runs the CER/WER
evaluator, and appends one row to:

```text
inference/results/sequential.csv
inference/results/batched.csv
```

Start exactly one server configuration first. The final optimized server is:

```bash
bash inference/run_vllm_fp8_static_audio_encoder_cudagraphs.sh
```

Keep the server `PORT` and benchmark `--base-url` consistent. Both currently
default to port 8090.

Use a configuration-specific output root. `analyse_results.py` derives the
precision label from this path, so a misleading output name produces a
misclassified result row.

The canonical matrix is:

```bash
# Batched: 100 measured clips limited to 50 seconds after 100 warm-ups.
uv run inference/run_benchmark.py \
  --mode batched \
  --workers 16 \
  --overwrite \
  --uniform-audio-length 50 \
  --num-files 100 \
  --warmup-files 100

# Batched: complete natural workload after the default 20 warm-ups.
uv run inference/run_benchmark.py \
  --mode batched \
  --workers 16 \
  --overwrite

# Sequential: 100 measured clips limited to 50 seconds after 100 warm-ups.
uv run inference/run_benchmark.py \
  --mode sequential \
  --num-files 100 \
  --uniform-audio-length 50 \
  --warmup-files 100

# Sequential: complete natural workload after the default 20 warm-ups.
uv run inference/run_benchmark.py \
  --mode sequential \
  --overwrite
```

Either update `DEFAULT_BATCHED_OUTPUT_ROOT` and
`DEFAULT_SEQUENTIAL_OUTPUT_ROOT` for the running configuration or pass an
explicit path:

```bash
uv run inference/run_benchmark.py \
  --mode batched \
  --workers 16 \
  --overwrite \
  --output-root predictions/results_fp8_static_audio_encoder_cudagraphs/batched_predicted
```

CER/WER runs only when all three conditions hold:

```text
--overwrite is set
--num-files is omitted
--uniform-audio-length is omitted
```

The fixed-50 runs therefore measure performance but are intentionally not
scored. The full runs score the completed prediction tree against
`data/prepared_data`.

Important options:

| Option | Meaning |
|---|---|
| `--mode sequential|batched` | Select the client implementation. |
| `--workers N` | Concurrent request workers; batched mode only. |
| `--warmup-files N` | Untimed requests before measured inference. |
| `--num-files N` | Limit measured inputs; disables automatic CER/WER. |
| `--uniform-audio-length S` | Clip eligible inputs to `S` seconds. |
| `--overwrite` | Write predictions; required for automatic scoring. |
| `--no-stream` | Measure non-streaming responses; TTFT is unavailable. |
| `--output-root PATH` | Isolate predictions and identify the precision. |
| `--base-url URL` | Select the running OpenAI-compatible server. |

### `analyse_results.py`

Render the accumulated comparison tables with:

```bash
uv run inference/analyse_results.py --mode batched
uv run inference/analyse_results.py --mode sequential
```

It prints p50/p95/p99 latency and TTFT, throughput, and full-workload CER/WER.
The full table selects rows with exactly 550 completed files and no uniform
audio limit. The fixed table selects rows whose uniform limit is 50 seconds.

Sequential mode prints the newest matching row for each recognized precision.
Batched mode prints every matching row, ordered by precision, worker count, and
timestamp. If several batched experiments share the same output-root precision
token, several rows can appear.

Recognized output-root tokens are:

```text
bf16
fp8_dynamic
fp8_static
fp8_static_qk_mrope_kv_cache_fusion
fp8_static_audio_encoder_cudagraphs
```

## CUDA correctness and microbenchmark helpers

These helpers create synthetic tensors and deterministic synthetic weights.
They do not decode audio, load the production checkpoint, start an HTTP server,
or calculate CER/WER.

Timing fields ending in `_cuda_us` measure GPU completion with CUDA events.
Fields ending in `_host_us` or `_host_call_us` primarily measure Python and
CUDA enqueue overhead. A positive reported speedup means:

```text
eager_time / candidate_time - 1 > 0
```

### CPU max-seqlen helper

File: `inference/vllm_qwen3_asr_optimizations/benchmarks/bench_audio_cpu_maxseqlen.py`

This checks that using the safe CPU upper bound `104` produces FlashAttention
outputs within tolerance of the dynamically computed GPU maximum. It then
measures the cost of calling `.item()` on the GPU scalar versus the CPU scalar
across the 24 audio-transformer layers.

```bash
CUDA_VISIBLE_DEVICES=<free-gpu> \
uv run inference/vllm_qwen3_asr_optimizations/benchmarks/bench_audio_cpu_maxseqlen.py \
  --repeats 200 \
  --layers 24 \
  --heads 16 \
  --head-dim 64
```

Check that all three length-layout cases complete without an assertion. The
last lines report:

```text
gpu_scalar_item_us_per_layer=...
cpu_scalar_item_us_per_layer=...
host_item_saved_us_per_24_layers=...
```

This helper uses tolerance-based attention comparison, not bitwise equality.

### CPU metadata and valid-row-pack helper

File: `inference/vllm_qwen3_asr_optimizations/benchmarks/bench_audio_cpu_metadata_pack.py`

This compares the original GPU-length, boolean-index reference with the
CPU-built metadata and Triton valid-row pack. It requires bitwise output
equality, profiles synchronization events, and reports median host and CUDA
time for both paths.

```bash
CUDA_VISIBLE_DEVICES=<free-gpu> \
uv run inference/vllm_qwen3_asr_optimizations/benchmarks/bench_audio_cpu_metadata_pack.py \
  --chunks 448 \
  --rows 13 \
  --hidden-size 1024 \
  --repeats 30
```

The first success line must be:

```text
correctness=exact output_shape=(..., 1024)
```

Compare `reference_profiler_sync_events` with
`candidate_profiler_sync_events`. The candidate is intended to remove the
GPU-to-host synchronization events in the reference metadata construction.

### Audio frontend CUDA graph helper

File: `inference/vllm_qwen3_asr_optimizations/benchmarks/bench_audio_frontend_cudagraph.py`

The audio frontend is:

```text
chunked/padded log-mel features
  -> three CNNs
  -> conv_out
  -> positional addition
  -> valid-row pack
  -> [M, 1024]
```

By default the helper validates all natural rows `M=377..390`. For each row it
admits endpoint feature-length aliases, compares graph replay bitwise with
eager execution, checks captured kernel order, verifies independent returned
storage, and exercises two-thread/two-stream replay. It also reports graph
memory consumption.

```bash
CUDA_VISIBLE_DEVICES=<free-h100> \
VLLM_LOGGING_LEVEL=WARNING \
uv run inference/vllm_qwen3_asr_optimizations/benchmarks/bench_audio_frontend_cudagraph.py \
  --warmup 3 \
  --repeats 10 \
  --replay-checks 3 \
  --concurrency-iterations 20
```

For a targeted row, repeat `--rows` as needed:

```bash
uv run inference/vllm_qwen3_asr_optimizations/benchmarks/bench_audio_frontend_cudagraph.py \
  --rows 377 \
  --rows 390 \
  --warmup 3 \
  --repeats 3 \
  --replay-checks 2 \
  --concurrency-iterations 2
```

Final acceptance marker:

```text
gate=PASS_EXACT_NATURAL_AUDIO_FRONTEND_CUDAGRAPH
```

### Audio transformer CUDA graph helper

File: `inference/vllm_qwen3_asr_optimizations/benchmarks/bench_audio_transformer_cudagraph.py`

The audio transformer is:

```text
hidden states [M, 1024]
  -> 24 audio-transformer layers
  -> ln_post -> proj1 -> activation -> proj2
  -> [M, 2048]
```

The default run validates all 14 canonical attention layouts
`(0,104,208,312,M)` for `M=377..390`. They share one padded 390-row graph.
Each exact key must complete seven eager probation observations and pass its
observation-eight bitwise gate. The helper also checks kernel order, retained
outputs, alternating cross-key reuse, and two-thread/two-stream replay.

```bash
CUDA_VISIBLE_DEVICES=<free-h100> \
VLLM_LOGGING_LEVEL=WARNING \
uv run inference/vllm_qwen3_asr_optimizations/benchmarks/bench_audio_transformer_cudagraph.py \
  --warmup 3 \
  --repeats 10 \
  --concurrency-iterations 20 \
  --master-port 29617
```

A targeted run still needs at least two distinct keys because the alternating
and concurrency gates require two requests sharing one bucket:

```bash
uv run inference/vllm_qwen3_asr_optimizations/benchmarks/bench_audio_transformer_cudagraph.py \
  --segments 104,104,104,65 \
  --segments 104,104,104,78 \
  --warmup 3 \
  --repeats 3 \
  --concurrency-iterations 2 \
  --master-port 29617
```

Those cases are `M=377` and `M=390`. Final acceptance marker:

```text
gate=PASS_BUCKETED_AUDIO_TRANSFORMER_CUDAGRAPH
```

### Combined audio-encoder helper

File: `inference/vllm_qwen3_asr_optimizations/benchmarks/bench_audio_encoder_cudagraphs.py`

This runs the two graph caches as one audio-encoder chain. It verifies that
calls 1–7 remain eager, call 8 admits both caches, changed input content remains
bitwise exact, and concurrent chained replay is safe. It then compares complete
eager audio-encoder execution with input copy, both graph replays, and
output cloning.

The default representative is `M=384`; `--rows` accepts one value from
`377..390`.

```bash
CUDA_VISIBLE_DEVICES=<free-h100> \
VLLM_LOGGING_LEVEL=WARNING \
uv run inference/vllm_qwen3_asr_optimizations/benchmarks/bench_audio_encoder_cudagraphs.py \
  --rows 384 \
  --warmup 3 \
  --repeats 10 \
  --concurrency-iterations 20 \
  --master-port 29619
```

Final acceptance marker:

```text
correctness=bitwise_exact
probation=PASS observations=8
concurrency=PASS threads=2 streams=2
gate=PASS_EXACT_AUDIO_ENCODER_CUDAGRAPHS
```

## Nsight Systems workflow

The Nsight scripts profile the server process, not the HTTP client.

### Terminal 1: launch the server under Nsight

Edit these variables in `inference/start_vllm_server_with_nsys.sh`:

```bash
SESSION_NAME=fp8static_qk_kvcache_fuse_c1
REPORT_NAME=fp8static_qk_kvcache_fuse_c1_5s
VLLM_SCRIPT=inference/run_vllm_fp8_static_qk_mrope_kv_cache_fusion.sh
```

Choose one trace mode:

```bash
bash inference/start_vllm_server_with_nsys.sh node

# Or CUDA graph launches plus Python/PyTorch ownership:
bash inference/start_vllm_server_with_nsys.sh pytrace_graph
```

`node` records executed CUDA graph nodes and kernel order. `pytrace_graph`
records graph launches with Python/PyTorch function tracing. Reports are
written under `inference/results/nsys/` as `.nsys-rep` and SQLite artifacts.

### Terminal 2: warm up and capture one request

Set matching values in `inference/run_nsys_profile.sh`:

```bash
SESSION=fp8static_qk_kvcache_fuse_c1_node
REPORT=inference/results/nsys/fp8static_qk_kvcache_fuse_c1_5s_node
AUDIO=data/prepared_data/.../channel_0.wav
```

Then run:

```bash
bash inference/run_nsys_profile.sh
```

The client waits for port 8090, performs three unprofiled five-second warm-ups,
starts collection, sends one measured five-second streaming request, and stops
the session. `SESSION` must exactly match the mode-specific session created by
the server script.

For trace interpretation and SQLite queries, see
[Nsight Systems guide](nsys-fp8-optimization-guide.md).

## Recommended validation order

For a change spanning the final audio stack:

1. Run the focused unit tests for the changed patch.
2. Run the CPU max-seqlen helper if attention metadata changed.
3. Run the CPU metadata-pack helper if chunking or valid-row packing changed.
4. Run the frontend or transformer helper for the modified graph cache.
5. Run the combined helper when either graph's interface or concurrency changes.
6. Run the four-command end-to-end matrix and inspect both analyzer tables.
7. Capture Nsight only when kernel order or unexplained latency needs tracing.

Promotion requires all relevant exactness markers, zero failed/timed-out
service requests, improved service metrics, and no material CER/WER regression.

## Common mistakes

- Do not interpret synthetic helper speedups as TTFT or request-latency gains.
- Do not run graph helpers on a non-SM90 GPU.
- Do not reuse an occupied NCCL `--master-port`.
- Do not benchmark on a GPU shared with another server or job.
- Do not compare fixed-50 rows with different warm-up counts without saying so.
- Do not use the same prediction output root for different precision paths.
- Do not expect CER/WER when `--num-files` or `--uniform-audio-length` is set.
- Do not profile only the HTTP client; Nsight must own the vLLM server process.
