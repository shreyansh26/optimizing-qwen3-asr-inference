# Optimizing Qwen3-ASR Inference

This repository is an end-to-end performance and quality harness for serving
`Qwen/Qwen3-ASR-1.7B` with vLLM. It prepares matched audio/transcript data,
runs sequential and concurrent inference, measures latency, time to first token
(TTFT), and throughput, evaluates CER/WER, and captures Nsight Systems traces
for kernel-level optimization work.

The supported precision paths are BF16, vLLM dynamic FP8, calibrated
static-activation FP8, and an optimized static-FP8 stack with custom Triton
kernels for Q/K RMSNorm, MRoPE, paged KV-cache writes, CPU-built audio
metadata, an exact valid-row pack, and guarded audio-encoder CUDA graph caches.

> **Current optimized path:**
> `PORT=8091 bash inference/run_vllm_fp8_static_qk_prefill_audio_prefix_suffix_cudagraph.sh`
>
> The final natural-only graph policy admits canonical 29--30 second server
> chunks and leaves arbitrary final tails on the general eager path. In the
> refreshed 550-file, 16-worker benchmark it reached `5.722 files/s`, latency
> p50/p95/p99 of `2.521/4.830/6.653 s`, and TTFT p50/p95/p99 of
> `0.543/1.328/1.759 s`, with CER `0.160` and WER `0.382`. Relative to the
> fused static-FP8 decoder control, throughput improved `24.28%` and latency
> percentiles improved `19.64--23.04%`; TTFT p50/p95 regressed while p99 was
> effectively unchanged. See [Current results](#current-results) and the
> [final CUDA-graph report](docs/audio-natural-only-cudagraph-benchmark.md).

All commands below assume the repository root:

```bash
cd /mnt/ssd1/shreyansh/home_dir/asr_experiments
```

## Contents

- [How the workflow fits together](#how-the-workflow-fits-together)
- [Quick start](#quick-start)
- [Data and artifact layout](#data-and-artifact-layout)
- [Prepare data](#prepare-data)
- [Direct inference](#direct-inference)
  - [Sequential inference](#sequential-inference)
  - [Concurrent inference](#concurrent-inference)
- [Benchmark and compare configurations](#benchmark-and-compare-configurations)
  - [Precision comparison workflow](#precision-comparison-workflow)
  - [Current results](#current-results)
- [Profile with Nsight Systems](#profile-with-nsight-systems)
- [Evaluate CER and WER](#evaluate-cer-and-wer)
- [End-to-end recipes](#end-to-end-recipes)
- [Technical documentation](#technical-documentation)

## How the workflow fits together

```text
manifest + source audio
          |
          v
  data/prepared_data  -->  vLLM server  -->  prediction tree
                                 |                 |
                                 v                 v
                         latency / TTFT /       CER / WER
                           throughput
```

The prepared reference tree and each prediction tree share the same relative
paths. This lets the evaluator match transcripts without a separate index and
allows failed or partial inference runs to be scored safely.

## Quick start

Install the pinned environment with `uv`:

```bash
uv sync
```

Start one vLLM server in the first terminal. The latency-optimized batched
static-FP8 launcher is the recommended starting point:

```bash
PORT=8091 \
  bash inference/run_vllm_fp8_static_qk_prefill_audio_prefix_suffix_cudagraph.sh
```

In a second terminal, run the two standard batched benchmarks and print the
aggregate comparison table:

```bash
# Controlled 100-file workload with 50-second clips.
uv run inference/run_benchmark.py \
  --mode batched \
  --workers 16 \
  --overwrite \
  --uniform-audio-length 50 \
  --num-files 100 \
  --input-root data/prepared_data \
  --warmup-files 100 \
  --output-root predictions/results_fp8_static_qk_prefill_audio_prefix_suffix_cudagraph_no_tail/batched_predicted_uniform_audio_length_50s

# Full workload; also computes aggregate CER/WER.
uv run inference/run_benchmark.py \
  --mode batched \
  --workers 16 \
  --overwrite \
  --input-root data/prepared_data \
  --warmup-files 20 \
  --output-root predictions/results_fp8_static_qk_prefill_audio_prefix_suffix_cudagraph_no_tail/batched_predicted

uv run inference/analyse_results.py --mode batched
```

`run_benchmark.py` currently defaults to the quick-start server at:

```text
http://localhost:8091/v1
```

The direct `run_infer.py`/`run_infer_batched.py` clients and most launchers
default to port 8090. Keep `PORT` and `--base-url` paired when using a different
launcher or client combination.

Choose one server precision:

```bash
# BF16.
bash inference/run_vllm.sh

# vLLM dynamic FP8 activations.
bash inference/run_vllm_fp8_dynamic.sh

# Calibrated static FP8 activations. The default artifact must exist.
bash inference/run_vllm_fp8_static.sh

# Static FP8 plus fused Q/K RMSNorm, MRoPE, and KV-cache update.
bash inference/run_vllm_fp8_static_qk_prefill.sh

# Current best: CPU metadata plus natural-only prefix and suffix graph caches.
bash inference/run_vllm_fp8_static_qk_prefill_audio_prefix_suffix_cudagraph.sh
```

The inference and benchmark scripts reset vLLM prefix cache before each run.
All launchers enable the vLLM development API endpoints needed for that reset.
Do not assume multimodal or encoder reset endpoints exist unless the running
server has been checked for them.

The static-FP8 launcher defaults to
`inference/results/fp8_static_scales_128x50.json`. Generate it with:

```bash
bash inference/run_record_fp8_static_scales.sh \
  --input-root data/prepared_data \
  --num-files 128 \
  --batch-size 8 \
  --max-audio-seconds 50 \
  --output inference/results/fp8_static_scales_128x50.json
```

Override the artifact when launching:

```bash
SCALES_JSON=inference/results/fp8_static_scales_custom.json \
  bash inference/run_vllm_fp8_static.sh
```

If the server runs elsewhere, pass `--base-url`.

For a controlled precision comparison, start exactly one server at a time and
give every configuration its own output root. The complete comparison matrix is
documented under [Precision comparison workflow](#precision-comparison-workflow).

## Data and artifact layout

Default input and output locations:

```text
data/manifest.json                    # source manifest mapping for data prep
data/prepared_data/                   # default ground-truth audio/text tree
predictions/results/sequential_predicted/            # default sequential prediction tree
predictions/results/batched_predicted/               # default batched prediction tree
predictions/results/sequential_predicted_uniform_audio_length_* # default sequential clipped trees
predictions/results/batched_predicted_uniform_audio_length_*    # default batched clipped trees
predictions/results_bf16/                 # BF16 comparison benchmark outputs
predictions/results_fp8_dynamic/          # dynamic-FP8 comparison benchmark outputs
predictions/results_fp8_static/           # static-FP8 comparison/default wrapper outputs
predictions/results_fp8_static_qk_prefill/ # static-FP8 Q/K fusion benchmark outputs
predictions/results_fp8_static_qk_prefill_audio_prefix_suffix_cudagraph_no_tail/
                                             # current natural-only graph outputs
```

Prepared data and predictions use the same relative path layout, for example:

```text
data/prepared_data/<dataset>/<sample>/channel_0.wav
data/prepared_data/<dataset>/<sample>/channel_0.txt
predictions/results/batched_predicted/<dataset>/<sample>/channel_0.txt
```

## Prepare data

Entry point:

```bash
uv run python data/read_data.py
```

Current fixed behavior:

- Reads `data/manifest.json`.
- Writes to `data/prepared_data`.
- Copies audio files.
- Overwrites existing prepared audio/text files.

This script does not currently expose CLI flags. To change the manifest,
output directory, placement mode, or overwrite behavior, edit the `prepare_dataset`
call at the bottom of `data/read_data.py`.

Supported placement modes in code:

- `copy`: copy audio files into prepared data.
- `symlink`: symlink prepared audio files to source audio files.
- `hardlink`: hardlink source audio files.
- `auto`: try hardlink first, then symlink.

## Direct inference

The low-level runners are useful for smoke tests, debugging, and custom jobs.
For comparable experiment records, prefer the
[benchmark wrapper](#benchmark-and-compare-configurations), which invokes these
runners and writes their metrics to CSV.

### Sequential inference

Entry point:

```bash
uv run python inference/run_infer.py <audio_path>
```

Single-file example:

```bash
uv run python inference/run_infer.py data/prepared_data/carta_september_2024/example/channel_0.wav
```

Arguments:

```bash
uv run python inference/run_infer.py <audio_path> \
  --model Qwen/Qwen3-ASR-1.7B \
  --base-url http://localhost:8090/v1
```

Streaming mode:

```bash
uv run python inference/run_infer.py <audio_path> --stream
```

Sequential directory mode:

```bash
uv run python inference/run_infer.py \
  --input-root data/prepared_data \
  --num-files 10 \
  --stream \
  --no-print-text
```

Uniform clipping with RMS no-speech filtering:

```bash
uv run python inference/run_infer.py \
  --num-files 10 \
  --uniform-audio-length 10 \
  --no-speech-rms-threshold 1
```

Sequential streaming with clipping, RMS filtering, and no transcript printing:

```bash
uv run python inference/run_infer.py \
  --input-root data/prepared_data/carta_september_2024 \
  --num-files 10 \
  --uniform-audio-length 10 \
  --no-speech-rms-threshold 1 \
  --stream \
  --no-print-text
```

Default behavior:

- With `audio_path`, runs one file and prints transcript text by default.
- Without `audio_path`, scans `.wav` files under `--input-root`.
- Runs requests sequentially and writes predictions under `predictions/results/sequential_predicted`.
- Resets vLLM prefix cache before warmup and measured inference.
- Uses 20 warmup audio files before measured files in directory mode.
- `--num-files` means measured benchmark files after warmup and after filtering.
- If an output file exists, inference still runs; without `--overwrite`, the
  existing file is not rewritten. With `--overwrite`, the output is written.
- `--print-text` prints transcripts; `--no-print-text` suppresses them.
- Final output reports completed/failed counts, throughput, audio throughput,
  avg/p50/p95/p99 latency, and avg/p50/p95/p99 TTFT (`n/a` for non-streaming).

### Concurrent inference

Entry point:

```bash
uv run python inference/run_infer_batched.py
```

Default behavior:

- Reads `.wav` files under `data/prepared_data`.
- Writes predictions under `predictions/results/batched_predicted`.
- Uses model `Qwen/Qwen3-ASR-1.7B`.
- Uses base URL `http://localhost:8090/v1`.
- Uses `--workers 1`.
- Uses `--timeout-seconds 10`.
- Uses `--max-tokens 512`.
- Resets vLLM prefix cache before warmup and measured inference.
- Uses 20 warmup audio files before measured files.
- `--num-files` means measured benchmark files after warmup and after filtering.
- If an output file exists, inference still runs; without `--overwrite`, the
  existing file is not rewritten. With `--overwrite`, the output is written.
- Continues past per-request failures and reports `failed` and `timed_out`.

Common commands:

```bash
# Full run.
uv run python inference/run_infer_batched.py

# Full run, overwrite existing predictions.
uv run python inference/run_infer_batched.py --overwrite

# Parallel run.
uv run python inference/run_infer_batched.py --workers 4 --overwrite

# Longer per-request timeout.
uv run python inference/run_infer_batched.py --workers 4 --overwrite --timeout-seconds 30

# Limit generated transcription tokens.
uv run python inference/run_infer_batched.py --workers 4 --overwrite --max-tokens 512

# Smoke test on the first N measured files after warmup/filtering.
uv run python inference/run_infer_batched.py --num-files 20 --overwrite

# Use a custom server or model.
uv run python inference/run_infer_batched.py \
  --base-url http://localhost:8090/v1 \
  --model Qwen/Qwen3-ASR-1.7B

# Read from and write to custom directories.
uv run python inference/run_infer_batched.py \
  --input-root data/prepared_data \
  --output-root predictions/results/batched_predicted_custom \
  --workers 4 \
  --overwrite
```

#### Clipped-audio mode

Use `--uniform-audio-length <seconds>` to send only the first fixed-length clip
from each audio file. Files shorter than or equal to the clip length are skipped.

```bash
# Send only first 10 seconds of each eligible audio file.
uv run python inference/run_infer_batched.py \
  --uniform-audio-length 10 \
  --workers 4 \
  --overwrite
```

Default clipped output path is a separate sibling directory:

```text
predictions/results/batched_predicted_uniform_audio_length_10s
```

The clipped mode also applies a no-speech guard before creating inference jobs.
The default threshold is `--no-speech-rms-threshold 1`; clipped payloads with
RMS less than or equal to that value are skipped before submission to vLLM.

```bash
# Disable most no-speech filtering.
uv run python inference/run_infer_batched.py \
  --uniform-audio-length 10 \
  --no-speech-rms-threshold 0 \
  --workers 4 \
  --overwrite

# More aggressive no-speech filtering.
uv run python inference/run_infer_batched.py \
  --uniform-audio-length 10 \
  --no-speech-rms-threshold 5 \
  --workers 4 \
  --overwrite

# Override clipped output directory.
uv run python inference/run_infer_batched.py \
  --uniform-audio-length 10 \
  --output-root predictions/results/batched_predicted_10s_custom \
  --workers 4 \
  --overwrite
```

All load-test options:

```bash
uv run python inference/run_infer_batched.py \
  --input-root data/prepared_data \
  --output-root predictions/results/batched_predicted \
  --model Qwen/Qwen3-ASR-1.7B \
  --base-url http://localhost:8090/v1 \
  --workers 4 \
  --num-files 100 \
  --uniform-audio-length 10 \
  --timeout-seconds 30 \
  --max-tokens 512 \
  --no-speech-rms-threshold 1 \
  --overwrite
```

## Benchmark and compare configurations

`run_benchmark.py` is the canonical experiment entry point. It keeps warmup,
request execution, metric parsing, result recording, and full-run quality
evaluation consistent across server configurations.

Commands:

```bash
uv run python inference/run_benchmark.py --mode sequential
uv run python inference/run_benchmark.py --mode batched
```

The wrapper runs `run_infer.py` or `run_infer_batched.py`, streams the child
script output live, parses the final metrics, and appends one row to:

```text
inference/results/sequential.csv
inference/results/batched.csv
```

Default behavior:

- `--mode sequential` uses
  `predictions/results/sequential_predicted`.
- `--mode batched` uses `predictions/results/batched_predicted`.
- For precision comparisons, pass an explicit `--output-root` containing
  `results_bf16`, `results_fp8_dynamic`, `results_fp8_static`, or
  `results_fp8_static_qk_prefill`. The analysis script uses those path markers
  to identify the precision.
- Streaming is enabled by default; use `--no-stream` for non-streaming requests.
- `--workers` is only valid with `--mode batched`; default is `1`.
- `--num-files` is measured files after the 20-file warmup and after filtering.
- `--uniform-audio-length` filters eligible audio first, then clips submitted
  audio to that length.
- `--overwrite` writes outputs even when prediction files already exist. Without
  it, inference still runs but existing outputs are not rewritten.
- `--no-speech-rms-threshold` defaults to `1`.
- `--input-root`, `--model`, and `--base-url` default to the same values as the
  underlying scripts. The wrapper's default output roots are the static-FP8
  paths listed above.

CSV metrics include counts, skipped/no-speech counts, wall time, file
throughput, audio throughput, avg/p50/p95/p99 latency, avg/p50/p95/p99 TTFT,
and the exact command that was run. For full overwrite benchmarks, the wrapper
also runs aggregate error-rate scoring and appends CER/WER fields to the CSV.
That scoring step runs only when all of these are true:

- `--overwrite` is passed.
- `--num-files` is not passed.
- `--uniform-audio-length` is not passed.

The scoring step uses `eval/compute_error_rates.py <output_root> --ref-root
<input_root>` and does not pass `--per-file`.

Common benchmark commands:

```bash
# Sequential streaming benchmark on 50 measured files.
uv run python inference/run_benchmark.py \
  --mode sequential \
  --num-files 50

# Sequential streaming benchmark with 10-second clipped audio.
uv run python inference/run_benchmark.py \
  --mode sequential \
  --num-files 50 \
  --uniform-audio-length 10

# Batched streaming benchmark with 4 concurrent workers.
uv run python inference/run_benchmark.py \
  --mode batched \
  --workers 4 \
  --num-files 50

# Batched 10-second clipped benchmark.
uv run python inference/run_benchmark.py \
  --mode batched \
  --workers 4 \
  --num-files 50 \
  --uniform-audio-length 10

# Non-streaming benchmark.
uv run python inference/run_benchmark.py \
  --mode batched \
  --workers 4 \
  --num-files 50 \
  --no-stream

# Write or overwrite prediction files during benchmarking.
uv run python inference/run_benchmark.py \
  --mode batched \
  --workers 4 \
  --num-files 50 \
  --overwrite

# Full batched benchmark with aggregate CER/WER scoring.
uv run python inference/run_benchmark.py \
  --mode batched \
  --workers 4 \
  --overwrite

# Custom roots, model, and server.
uv run python inference/run_benchmark.py \
  --mode sequential \
  --input-root data/prepared_data \
  --output-root predictions/results/sequential_predicted_custom \
  --model Qwen/Qwen3-ASR-1.7B \
  --base-url http://localhost:8090/v1 \
  --num-files 50
```

### Precision comparison workflow

Start exactly one matching server, run its benchmarks, stop it, and repeat for
the next precision:

| Precision label | Server command |
| --- | --- |
| `bf16` | `bash inference/run_vllm.sh` |
| `fp8_dynamic` | `bash inference/run_vllm_fp8_dynamic.sh` |
| `fp8_static` | `bash inference/run_vllm_fp8_static.sh` |
| `fp8_static_qk_prefill` | `bash inference/run_vllm_fp8_static_qk_prefill.sh` |
| `fp8_static_qk_prefill_audio_prefix_suffix_cudagraph` | `bash inference/run_vllm_fp8_static_qk_prefill_audio_prefix_suffix_cudagraph.sh` |

Use precision-specific output roots. This keeps prediction files isolated and
allows `analyse_results.py` to infer the precision from the recorded path:

```bash
# Set this to the output-root label for the matching running server.
PRECISION=fp8_static_qk_prefill_audio_prefix_suffix_cudagraph_no_tail

# Full sequential benchmark; --overwrite also enables aggregate CER/WER scoring.
uv run python inference/run_benchmark.py \
  --mode sequential \
  --output-root "predictions/results_${PRECISION}/sequential_predicted" \
  --warmup-files 20 \
  --overwrite

# Sequential benchmark limited to the first 50 seconds of eligible files.
uv run python inference/run_benchmark.py \
  --mode sequential \
  --uniform-audio-length 50 \
  --num-files 100 \
  --warmup-files 100 \
  --overwrite \
  --output-root "predictions/results_${PRECISION}/sequential_predicted_uniform_audio_length_50s"

# Full 16-worker load test.
uv run python inference/run_benchmark.py \
  --mode batched \
  --workers 16 \
  --output-root "predictions/results_${PRECISION}/batched_predicted" \
  --warmup-files 20 \
  --overwrite

# 16-worker, 50-second load test.
uv run python inference/run_benchmark.py \
  --mode batched \
  --workers 16 \
  --uniform-audio-length 50 \
  --num-files 100 \
  --warmup-files 100 \
  --overwrite \
  --output-root "predictions/results_${PRECISION}/batched_predicted_uniform_audio_length_50s"
```

Print the comparison tables from the accumulated CSV files:

```bash
uv run python inference/analyse_results.py --mode sequential
uv run python inference/analyse_results.py --mode batched
```

### Current results

These are the rows currently selected from `inference/results/sequential.csv`
and `inference/results/batched.csv` by `analyse_results.py`. Latency and TTFT
values are seconds; throughput is completed files per second. The refreshed
final fixed-50 rows used 100 warm-up files, while most older fixed-50 controls
printed beside them used 20. The full rows used 20 warm-ups but were collected
on different dates and server lifetimes. Percentage deltas below are snapshot
comparisons, not matched causal A/B estimates or confidence intervals.
Generated predictions and appended CSV evidence remain local benchmark
artifacts rather than source-controlled fixtures.

`FP8 static + Q/K fusion` uses the dedicated launcher above. The optimization
runs in decode and prefill: it combines Q/K RMSNorm, three-axis MRoPE, and the
native BF16 paged KV-cache write, while inputs of 512 tokens or more use a
prefill-specific head grouping. See
[the technical design](docs/qk-mrope-kv-cache-fusion.md) for the execution flow,
specialization boundaries, and Nsight evidence.

The final row adds two audio-encoder layers. First, CPU length metadata and a
single Triton row pack remove synchronization-heavy GPU scalar/list
materialization; the CPU max-seqlen cache also removes the repeated scalar
readback in all 24 audio-transformer layers. Second, exact-admitted CUDA graph
caches replay the 29--30 second convolution/projection/pack prefix and the
24-layer/projection suffix. Graph-ineligible durations or metadata that still
satisfy the CPU runtime guard use CPU metadata plus eager prefix/suffix;
runtime-guard misses delegate to original vLLM. See the
[audio-length and fast-path guide](docs/qwen3-asr-audio-length-and-graph-fast-path.md)
for the complete data flow and admission contract.

Sequential, full benchmark with 550 measured files:

| Precision | Lat p50 | Lat p95 | Lat p99 | TTFT p50 | TTFT p95 | TTFT p99 | Throughput | CER | WER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 1.576 | 3.261 | 4.229 | 0.203 | 0.318 | 0.377 | 0.583 | 0.168 | 0.388 |
| FP8 dynamic | 1.473 | 2.929 | 3.732 | 0.201 | 0.309 | 0.339 | 0.640 | 0.165 | 0.385 |
| FP8 static | 1.381 | 2.745 | 3.439 | 0.207 | 0.339 | 0.372 | 0.682 | 0.162 | 0.384 |
| FP8 static + Q/K fusion | 1.315 | 2.599 | 3.275 | 0.220 | 0.341 | 0.373 | 0.716 | 0.163 | 0.385 |
| **FP8 static + Q/K + CPU metadata + natural audio graphs** | **1.214** | **2.494** | **3.110** | **0.203** | **0.301** | **0.329** | **0.763** | **0.160** | **0.383** |

Sequential, 100 measured files with a 50-second audio limit:

| Precision | Lat p50 | Lat p95 | Lat p99 | TTFT p50 | TTFT p95 | TTFT p99 | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 0.365 | 0.564 | 1.411 | 0.063 | 0.074 | 0.092 | 2.677 |
| FP8 dynamic | 0.286 | 0.479 | 0.634 | 0.046 | 0.060 | 0.064 | 3.409 |
| FP8 static | 0.273 | 0.468 | 0.489 | 0.059 | 0.075 | 0.090 | 3.538 |
| FP8 static + Q/K fusion | 0.262 | 0.451 | 0.473 | 0.063 | 0.087 | 0.091 | 3.731 |
| **FP8 static + Q/K + CPU metadata + natural audio graphs** | **0.202** | **0.333** | **0.398** | **0.050** | **0.063** | **0.077** | **4.551** |

Batched, full benchmark with 550 measured files:

| Precision | Workers | Lat p50 | Lat p95 | Lat p99 | TTFT p50 | TTFT p95 | TTFT p99 | Throughput | CER | WER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 4 | 1.977 | 4.006 | 5.276 | 0.227 | 0.402 | 0.509 | 1.875 | n/a | n/a |
| BF16 | 8 | 2.428 | 5.082 | 6.409 | 0.260 | 0.558 | 0.935 | 3.001 | n/a | n/a |
| BF16 | 16 | 3.520 | 7.136 | 9.130 | 0.351 | 0.946 | 1.946 | 4.138 | 0.166 | 0.387 |
| FP8 dynamic | 4 | 1.900 | 3.831 | 4.719 | 0.228 | 0.420 | 0.566 | 1.980 | n/a | n/a |
| FP8 dynamic | 8 | 2.435 | 4.959 | 5.848 | 0.270 | 0.525 | 0.775 | 3.055 | n/a | n/a |
| FP8 dynamic | 16 | 3.563 | 7.166 | 9.903 | 0.343 | 1.121 | 1.838 | 4.104 | 0.158 | 0.380 |
| FP8 static | 16 | 3.346 | 6.675 | 9.124 | 0.426 | 1.177 | 1.984 | 4.352 | 0.162 | 0.384 |
| FP8 static + Q/K fusion | 16 | 3.137 | 6.276 | 8.382 | 0.385 | 1.069 | 1.761 | 4.604 | 0.166 | 0.387 |
| **FP8 static + Q/K + CPU metadata + natural audio graphs** | **16** | **2.521** | **4.830** | **6.653** | 0.543 | 1.328 | **1.759** | **5.722** | **0.160** | **0.382** |

Batched, 16 workers and 100 measured files with a 50-second audio limit:

| Precision | Workers | Lat p50 | Lat p95 | Lat p99 | TTFT p50 | TTFT p95 | TTFT p99 | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 4 | 0.429 | 0.641 | 1.659 | 0.049 | 0.085 | 0.122 | 9.130 |
| BF16 | 8 | 0.542 | 0.864 | 2.047 | 0.073 | 0.179 | 0.288 | 13.922 |
| BF16 | 16 | 0.720 | 1.338 | 2.768 | 0.138 | 0.448 | 0.456 | 18.554 |
| FP8 dynamic | 4 | 0.351 | 0.582 | 0.727 | 0.052 | 0.106 | 0.111 | 10.489 |
| FP8 dynamic | 8 | 0.441 | 0.764 | 0.846 | 0.053 | 0.149 | 0.157 | 16.686 |
| FP8 dynamic | 16 | 0.735 | 1.279 | 1.481 | 0.154 | 0.432 | 0.499 | 20.731 |
| FP8 static | 16 | 0.652 | 1.098 | 1.212 | 0.119 | 0.306 | 0.332 | 22.171 |
| FP8 static + Q/K fusion | 16 | 0.636 | 1.226 | 1.424 | 0.122 | 0.499 | 0.614 | 22.635 |
| **FP8 static + Q/K + CPU metadata + natural audio graphs** | **16** | **0.426** | **0.890** | **1.008** | **0.086** | **0.216** | **0.227** | **28.192** |

Relative to the displayed `FP8 static + Q/K fusion` snapshot, the final
16-worker full-workload row
improves throughput by `24.28%` and latency p50/p95/p99 by
`19.64%/23.04%/20.63%`. TTFT p50 and p95 regress by `41.04%` and `24.23%`,
while TTFT p99 improves by `0.11%`. On fixed-50 it improves throughput by
`24.55%`, latency percentiles by `27.41--33.02%`, and TTFT percentiles by
`29.51--63.03%`, even though only the natural first chunk is graphed and the
approximately 20--21 second tail remains eager.

The sequential improvements versus the displayed fused snapshot are smaller
on the varying full workload: `+6.56%` throughput, `4.04--7.68%` lower latency
percentiles, and `7.73--11.80%` lower TTFT percentiles. The fixed-50 sequential
row gains `21.98%` throughput. These comparisons are direct calculations from
the rounded rows printed by `analyse_results.py`, not confidence intervals.

Static FP8 currently has batched comparison rows only at 16 workers; the table
does not imply unmeasured 4- or 8-worker static results. CER/WER is `n/a` for
rows that were not recorded through the wrapper's full overwrite-and-score
path. The 50-second runs are intentionally not scored by the wrapper.

The refreshed final full rows report CER/WER `0.160/0.383` sequentially and
`0.160/0.382` in batched mode. They do not show a material quality regression
relative to the fused static-FP8 control (`0.163/0.385` sequential and
`0.166/0.387` batched). The graph admission checks also compare the capture
result bitwise with eager execution before enabling replay.

## Profile with Nsight Systems

Nsight Systems must launch the vLLM server so that CUDA work from the engine
process is visible. The client request then starts and stops collection through
the named Nsight session. Run the workflow from the repository root in two
terminals.

The two supported modes answer different questions:

| Mode | Nsight configuration | Use it for |
| --- | --- | --- |
| `node` | `--cuda-graph-trace=node` | Executed CUDA graph nodes, exact kernel order and names, node counts, and per-replay GPU timing |
| `pytrace_graph` | `--cuda-graph-trace=graph --pytorch=functions-trace` | CUDA graph launches plus Python/PyTorch/NVTX ownership and CPU call-path context |

Use `node` for kernel-level dynamic-versus-static FP8 comparisons. Use
`pytrace_graph` to determine which Python or vLLM region initiated a graph
launch. The server wrapper supplies the required Python `nvtx` package through
`uv run --with nvtx`, so this does not modify the project dependencies.

### Configure the server capture

Edit these variables near the top of
`inference/start_vllm_server_with_nsys.sh`:

```bash
SESSION_NAME=fp8static_c1
REPORT_NAME=fp8static_c1_5s
VLLM_SCRIPT=inference/run_vllm_fp8_static.sh
```

Common precision configurations are:

| Precision | `SESSION_NAME` | `REPORT_NAME` | `VLLM_SCRIPT` |
| --- | --- | --- | --- |
| BF16 | `bf16_c1` | `bf16_c1_5s` | `inference/run_vllm.sh` |
| Dynamic FP8 | `fp8dyn_c1` | `fp8dyn_c1_5s` | `inference/run_vllm_fp8_dynamic.sh` |
| Static FP8 | `fp8static_c1` | `fp8static_c1_5s` | `inference/run_vllm_fp8_static.sh` |
| Static FP8 + Q/K fusion | `fp8static_qk_kvcache_fuse_c1` | `fp8static_qk_kvcache_fuse_c1_5s` | `inference/run_vllm_fp8_static_qk_prefill.sh` |

The wrapper appends the selected mode to both names. For example, the static
configuration produces:

```text
node session:          fp8static_c1_node
node report:           inference/results/nsys/fp8static_c1_5s_node
pytrace_graph session: fp8static_c1_pytrace_graph
pytrace_graph report:  inference/results/nsys/fp8static_c1_5s_pytrace_graph
```

### Configure the profiled request

Edit the matching values near the top of `inference/run_nsys_profile.sh`:

```bash
SESSION=fp8static_c1_node
REPORT=inference/results/nsys/fp8static_c1_5s_node
AUDIO=data/prepared_data/carta_september_2024/call-156003_0.8311693678982316_4.230553711834489/channel_0.wav
```

`SESSION` must exactly equal `<SESSION_NAME>_<mode>` from the server wrapper.
`REPORT` should equal
`inference/results/nsys/<REPORT_NAME>_<mode>`. The server wrapper owns the
actual Nsight output path; the request script keeps `REPORT` only to print the
expected artifact at the end.

The request script waits for `/v1/models`, runs three unprofiled warmups, starts
collection, sends one streaming 5-second request, and stops collection.

### Capture node-level CUDA graph activity

For the static configuration above, use:

```bash
# Terminal 1: launch the server under an inactive Nsight session.
bash inference/start_vllm_server_with_nsys.sh node
```

After the server starts, run in another terminal:

```bash
# Terminal 2: warm up, profile one request, and stop collection.
bash inference/run_nsys_profile.sh
```

For dynamic FP8, first set the server variables to:

```bash
SESSION_NAME=fp8dyn_c1
REPORT_NAME=fp8dyn_c1_5s
VLLM_SCRIPT=inference/run_vllm_fp8_dynamic.sh
```

and set the request variables to:

```bash
SESSION=fp8dyn_c1_node
REPORT=inference/results/nsys/fp8dyn_c1_5s_node
```

Then run the same two commands.

### Capture Python/PyTorch ownership and graph launches

Change the request script to the matching `pytrace_graph` session and report:

```bash
SESSION=fp8static_c1_pytrace_graph
REPORT=inference/results/nsys/fp8static_c1_5s_pytrace_graph
```

Then run:

```bash
# Terminal 1.
bash inference/start_vllm_server_with_nsys.sh pytrace_graph

# Terminal 2, after the server is ready.
bash inference/run_nsys_profile.sh
```

For dynamic FP8, use `fp8dyn_c1_pytrace_graph` and
`fp8dyn_c1_5s_pytrace_graph` instead.

Each capture writes both files because the server wrapper passes
`--export=sqlite`:

```text
inference/results/nsys/<report-name>.nsys-rep
inference/results/nsys/<report-name>.sqlite
```

If the vLLM process remains active after report generation, stop it with
Ctrl-C in Terminal 1 before starting the next precision or trace mode. Do not
run two servers on port `8090` at the same time.

See the
[Nsight Systems FP8 optimization guide](docs/nsys-fp8-optimization-guide.md)
for direct `nsys` commands, timing definitions, Events View filtering, SQLite
queries, graph identification, and interpretation of the captured kernels.

## Evaluate CER and WER

Entry point:

```bash
uv run python eval/compute_error_rates.py <prediction_root>
```

The evaluator computes case-insensitive, whitespace-normalized:

- CER: character error rate.
- WER: word error rate.

It iterates over prediction `.txt` files and matches each prediction to the
same relative path under the reference root. This supports partial prediction
runs where some requests failed or timed out.

Default reference root:

```text
data/prepared_data
```

Commands:

```bash
# Score full-audio predictions.
uv run python eval/compute_error_rates.py predictions/results/batched_predicted

# Score clipped-audio predictions.
uv run python eval/compute_error_rates.py predictions/results/batched_predicted_uniform_audio_length_10s

# Use a custom reference tree.
uv run python eval/compute_error_rates.py predictions/results/batched_predicted_custom \
  --ref-root data/prepared_data

# Print per-file tab-separated metrics before the aggregate summary.
uv run python eval/compute_error_rates.py predictions/results/batched_predicted \
  --per-file

# Save per-file and aggregate output.
uv run python eval/compute_error_rates.py predictions/results/batched_predicted \
  --per-file > error_rates.tsv
```

Summary output includes:

- number of prediction text files found;
- number of files matched to references;
- number of missing references;
- aggregate CER with `char_edits/ref_chars`;
- aggregate WER with `word_edits/ref_words`.

## End-to-end recipes

Prepare data, run inference, score predictions:

```bash
uv run python data/read_data.py
uv run python inference/run_infer_batched.py --workers 4 --overwrite --timeout-seconds 30
uv run python eval/compute_error_rates.py predictions/results/batched_predicted
```

Run a short smoke test:

```bash
uv run python inference/run_infer_batched.py --num-files 20 --workers 4 --overwrite
uv run python eval/compute_error_rates.py predictions/results/batched_predicted
```

Run clipped 10-second inference and score it:

```bash
uv run python inference/run_infer_batched.py \
  --uniform-audio-length 10 \
  --workers 4 \
  --overwrite \
  --timeout-seconds 30

uv run python eval/compute_error_rates.py predictions/results/batched_predicted_uniform_audio_length_10s
```

## Technical documentation

The README is the operational entry point. These focused guides contain the
implementation details, profiling evidence, and upgrade boundaries behind the
optimized paths:

- [Static FP8 activation calibration](docs/fp8-calibration.md) explains sample
  selection, batched collection, global per-layer absmax aggregation, scale
  calculation, and the portable JSON artifact.
- [Out-of-tree vLLM static-FP8 extension](docs/vllm-static-fp8-extension.md)
  explains Python/vLLM registration, layer-name mapping, scale injection,
  runtime behavior, coverage diagnostics, and upgrade boundaries.
- [Static-FP8 Q/K MRoPE and KV-cache fusion](docs/qk-mrope-kv-cache-fusion.md)
  explains the specialized Triton kernel, vLLM integration, exact RMS
  reduction order, prefill dispatch, correctness checks, profiling evidence,
  and current benchmark results.
- [Qwen3-ASR audio lengths and CUDA-graph fast-path coverage](docs/qwen3-asr-audio-length-and-graph-fast-path.md)
  is the authoritative implementation guide for CPU metadata construction,
  exact Triton row packing, prefix/suffix boundaries, natural-shape admission,
  CUDA graph cache behavior, fallback handling, and raw-audio length mapping.
- [Final natural-only audio CUDA-graph benchmark](docs/audio-natural-only-cudagraph-benchmark.md)
  records the current `analyse_results.py` tables, warm-up policy, comparison
  to the fused decoder control, quality results, and interpretation.
- [Benchmarking and CUDA helper guide](docs/benchmarking-guide.md) documents the
  end-to-end benchmark matrix, result aggregation, all five synthetic CUDA
  helpers, success markers, requirements, and the two-terminal Nsight workflow.
- [CPU audio metadata and exact valid-row packing](ideas/audio_cpu_metadata_pack.md)
  preserves the original synchronization audit and round-three validation that
  motivated the metadata layer beneath the final graph caches.
- [Nsight Systems FP8 optimization guide](docs/nsys-fp8-optimization-guide.md)
  covers capture commands, report interpretation, CUDA-graph timing, exact
  node-level differences, screenshots, and row-by-row decoder pseudocode for
  dynamic FP8, static FP8, and the fused Q/K-MRoPE-cache path.
- [FlashAttention forward combine in Nsight Systems](docs/flashattention-forward-combine.md)
  explains the split-KV decode path, stable softmax recombination, and how to
  interpret `FlashAttnFwdCombine` in the node trace.
