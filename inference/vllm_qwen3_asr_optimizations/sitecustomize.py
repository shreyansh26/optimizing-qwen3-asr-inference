"""Load the local Qwen3-ASR optimizations before vLLM parses its CLI."""

import os


if os.environ.get("ASR_FP8_STATIC_SCALES_JSON"):
    import vllm_qwen3_asr_optimization_plugin  # noqa: F401
