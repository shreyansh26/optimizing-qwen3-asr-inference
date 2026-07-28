# Audio encoder CPU max-seqlen experiment

## Hypothesis

The Qwen3-ASR audio encoder computes `max_seqlen` as a CUDA scalar once, then
passes that same scalar through all 24 encoder layers. vLLM's FlashAttention
wrapper calls `.item()` in every layer. Returning a cached CPU scalar removes
those repeated device-to-host synchronization points without changing
`cu_seqlens` or the attention operation.

## Exact guard

The patch is enabled only by `ASR_AUDIO_CPU_MAXSEQLEN=1` and only for the exact
installed `Qwen3OmniMoeAudioEncoder` class with the production FlashAttention
backend, 50/800 window geometry, 128 mel bins, 1500 source positions, 24 layers
of width 1024 with 16 heads of width 64, and three expected stride-2
convolutions. Any mismatch calls the original method.

The upper bound is exact for a full group:

```text
100 raw frames -> ceil(/2) -> 50 -> 25 -> 13 post-CNN frames
800 / 100 = 8 windows per attention group
13 * 8 = 104
```

Every shorter tail is bounded by 104. The returned tensor is a single cached
CPU `torch.int32` scalar; the CUDA `cu_seqlens` and FlashAttention inputs remain
unchanged.

## Validation

CPU tests cover the environment gate, every compatibility guard, cached scalar
identity, fallback behavior, short tails from 1 through 100 raw frames,
idempotence, and integration with the installed vLLM class.

GPU validation is intentionally separate:

```bash
UV_PROJECT_ENVIRONMENT=/mnt/ssd1/shreyansh/home_dir/asr_experiments/.venv \
  uv run inference/vllm_static_fp8/benchmarks/bench_audio_cpu_maxseqlen.py
```

The helper compares original and candidate FlashAttention outputs for full and
short variable-length batches, then measures host `.item()` latency for the
GPU scalar versus the cached CPU scalar over 24 layers.

The service launcher is:

```bash
CUDA_VISIBLE_DEVICES=1 PORT=8091 \
  inference/experimental_launchers/run_vllm_fp8_static_qk_prefill_audio_cpu_maxseqlen.sh
```

End-to-end batched results are pending.
