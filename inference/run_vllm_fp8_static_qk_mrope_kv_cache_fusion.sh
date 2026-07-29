#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export ASR_QK_MROPE_KV_CACHE_FUSION=1

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3-ASR-1.7B}"
if [[ -z "${MODEL:-}" ]]; then
  HF_CACHE_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
  MODEL_CACHE_DIR="$HF_CACHE_HOME/hub/models--Qwen--Qwen3-ASR-1.7B"
  if [[ -f "$MODEL_CACHE_DIR/refs/main" ]]; then
    MODEL_REVISION="$(< "$MODEL_CACHE_DIR/refs/main")"
    MODEL_SNAPSHOT="$MODEL_CACHE_DIR/snapshots/$MODEL_REVISION"
    if [[ -f "$MODEL_SNAPSHOT/config.json" ]]; then
      export MODEL="$MODEL_SNAPSHOT"
      export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
      export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    fi
  fi
fi

echo "Q/K RMSNorm + MRoPE + KV-cache fusion: enabled"

exec "$SCRIPT_DIR/run_vllm_fp8_static.sh" \
  --served-model-name "$SERVED_MODEL_NAME" \
  "$@"
