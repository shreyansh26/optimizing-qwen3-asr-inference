#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export ASR_AUDIO_CPU_MAXSEQLEN=1
export ASR_AUDIO_CPU_METADATA_PACK=1

echo "Qwen3-ASR audio metadata: CPU lengths + fused valid-row pack"

exec "$SCRIPT_DIR/../run_vllm_fp8_static_qk_mrope_kv_cache_fusion.sh" "$@"
