# Static FP8 activation calibration

This document explains how this repository produces the per-layer activation
scales consumed by the static-FP8 vLLM extension. The calibration artifact is
portable; the model name or revision, calibration settings, and generated JSON
should travel together.

For how that JSON is injected into vLLM, see
[Out-of-tree vLLM static-FP8 extension](vllm-static-fp8-extension.md).

## Quick start

Run calibration from the repository root:

```bash
bash inference/run_record_fp8_static_scales.sh \
  --input-root data/prepared_data \
  --num-files 128 \
  --batch-size 8 \
  --max-audio-seconds 50 \
  --output inference/results/fp8_static_scales_128x50.json
```

The wrapper is the recommended entry point. It runs the recorder in an
isolated `uv` overlay with the versions required by the Transformers/Qwen-ASR
checkpoint layout:

- `qwen-asr==0.0.6`
- `transformers==4.57.6` through the `qwen-asr` dependency
- PyTorch `2.11.0+cu128`

The serving environment remains the repository's normal vLLM environment.
Calibration does not install these overlay packages into the project.

You can also calibrate explicit files:

```bash
bash inference/run_record_fp8_static_scales.sh \
  data/prepared_data/dataset_a/sample_1/channel_0.wav \
  data/prepared_data/dataset_b/sample_2/channel_1.wav \
  --batch-size 2 \
  --output inference/results/fp8_static_scales_custom.json
```

## What is measured

`record_fp8_static_scales.py` loads the BF16 Transformers/Qwen-ASR model and
registers a forward-pre-hook on every `torch.nn.Linear` except `lm_head`. A
forward-pre-hook sees the input tensor that is about to be consumed by that
linear layer.

For every invocation of a layer, the recorder computes one scalar:

```text
call_absmax = max(abs(layer_input))
```

The maximum covers the entire input tensor: batch items, tokens, and hidden
features. The recorder then keeps the maximum over every call and every
calibration batch:

```text
final_absmax[layer] = max(call_absmax_1, call_absmax_2, ...)
```

This is a maximum, not an average. Batching changes how many examples are run
together, but the exported statistic remains the largest value observed for
that layer across the whole calibration run.

The FP8 E4M3 activation scale is:

```text
scale[layer] = max(
    final_absmax[layer] * scale_margin / 448,
    1 / (448 * 512),
)
```

The default `--scale-margin 1.05` adds 5% headroom above the largest observed
activation. `448` is the maximum finite magnitude used for FP8 E4M3FN. The
minimum scale prevents a zero or extremely small scale for nearly-zero
activations.

At serving time, the extension uses the scalar approximately as:

```text
x_fp8 = cast_fp8(x / scale[layer])
```

Values outside the calibrated range can saturate. That is why calibration data
should represent the audio durations, languages, acoustic conditions, and
request shapes expected in production.

## Audio selection and batching

When positional audio paths are not provided, the recorder:

1. Finds WAV files below `--input-root`.
2. Shuffles them deterministically with `--seed` (default `0`).
3. Selects the first `--num-files` files (default `128`).
4. When `--batch-size` is greater than one, sorts the selected set by clipped
   duration to reduce padding within batches.

Each file is:

- decoded as float32;
- mixed down to mono by averaging channels;
- limited to `--max-audio-seconds`;
- resampled to 16 kHz when necessary.

The default batch size is `8`. The model runs normal transcription, including
decoder generation up to `--max-new-tokens`, so hooks observe both audio-tower
and language-model linear inputs across the full ASR path.

## Mapping Transformers layers to vLLM fused layers

The Transformers model exposes separate projection modules that vLLM packs
into one linear layer. The artifact therefore contains both the raw
Transformers measurements in `modules` and vLLM-compatible entries in
`vllm_fused_modules`.

The recorder applies these mappings:

| Transformers projections | vLLM fused entry |
| --- | --- |
| `q_proj`, `k_proj`, `v_proj` | `qkv_proj` |
| `gate_proj`, `up_proj` | `gate_up_proj` |

For a packed layer, the exported scale is the maximum scale of its source
modules. Using the maximum gives the packed input one range large enough for
all constituent projections.

## Artifact contents

The JSON records enough metadata to audit or reproduce the calibration:

- model and optional revision;
- calibration backend and aggregation rule;
- FP8 dtype, maximum, minimum scale, and margin;
- batch size, audio-duration limit, generation limit, and random seed;
- Torch, Transformers, and Qwen-ASR versions;
- every selected audio path;
- raw per-linear statistics;
- vLLM fused-module scales.

The current `fp8_static_scales_128x50.json` artifact was produced with:

| Setting | Value |
| --- | ---: |
| Audio files | 128 |
| Batch size | 8 |
| Maximum audio duration | 50 seconds |
| Maximum new tokens | 512 |
| Scale margin | 1.05 |
| Raw linear modules | 343 |
| vLLM fused-module entries | 211 |

Inspect an artifact without loading the model:

```bash
jq '{
  format,
  version,
  model,
  scale_margin,
  batch_size,
  max_audio_seconds,
  audio_files: (.audio_files | length),
  modules: (.modules | length),
  vllm_fused_modules: (.vllm_fused_modules | length)
}' inference/results/fp8_static_scales_128x50.json
```

## Recalibration guidance

Recalibrate when any of these materially changes:

- model checkpoint or revision;
- expected audio-duration distribution;
- language or domain distribution;
- preprocessing;
- model implementation that changes module inputs;
- unacceptable clipping or accuracy drift.

Use a stable, representative calibration set rather than benchmark test cases
selected specifically for good results. Record the exact model revision when
reproducibility matters.

After calibration, start the server with the artifact explicitly:

```bash
SCALES_JSON=inference/results/fp8_static_scales_128x50.json \
  bash inference/run_vllm_fp8_static.sh
```

Then validate both performance and CER/WER. Static scales remove runtime scale
reductions, but prefix coverage only proves that every expected layer received
a scale; it does not prove that the calibration distribution is representative.

## Files involved

- `inference/record_fp8_static_scales.py`: audio selection, hooks, aggregation,
  scale calculation, fusion, and JSON export.
- `inference/run_record_fp8_static_scales.sh`: isolated dependency overlay for
  the calibration model.
- `inference/results/fp8_static_scales_128x50.json`: current portable scale
  artifact.
- `inference/vllm_qwen3_asr_optimizations/vllm_qwen3_asr_optimization_plugin.py`: consumer of
  `vllm_fused_modules` during vLLM startup.
