#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-ASR-1.7B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8090}"
SCALES_JSON="${SCALES_JSON:-$SCRIPT_DIR/results/fp8_static_scales_128x50.json}"

if [[ ! -f "$SCALES_JSON" ]]; then
  echo "Static FP8 scale JSON not found: $SCALES_JSON" >&2
  exit 1
fi

export ASR_FP8_STATIC_SCALES_JSON="$(realpath "$SCALES_JSON")"
DEFAULT_COVERAGE_JSON="/tmp/asr_fp8_static_coverage_${PORT}_{pid}.json"
export ASR_FP8_STATIC_COVERAGE_JSON="${ASR_FP8_STATIC_COVERAGE_JSON:-$DEFAULT_COVERAGE_JSON}"
export PYTHONPATH="$SCRIPT_DIR/vllm_qwen3_asr_optimizations${PYTHONPATH:+:$PYTHONPATH}"

echo "Model: $MODEL"
echo "Static FP8 scales: $ASR_FP8_STATIC_SCALES_JSON"
echo "Static FP8 coverage: $ASR_FP8_STATIC_COVERAGE_JSON"
echo "Serving on $HOST:$PORT"

VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-1}" \
  exec uv run vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --quantization fp8_static_json \
    "$@"
