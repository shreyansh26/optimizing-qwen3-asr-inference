#!/usr/bin/env bash
set -euo pipefail

cd /mnt/ssd1/shreyansh/home_dir/asr_experiments
mkdir -p inference/results/nsys

SESSION_NAME=fp8static_qk_kvcache_fuse_c1
REPORT_NAME=fp8static_qk_kvcache_fuse_c1_5s
VLLM_SCRIPT=inference/run_vllm_fp8_static_qk_mrope_kv_cache_fusion.sh

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {node|pytrace_graph}" >&2
  exit 2
fi

MODE=$1
SESSION="${SESSION_NAME}_${MODE}"
REPORT="/mnt/ssd1/shreyansh/home_dir/asr_experiments/inference/results/nsys/${REPORT_NAME}_${MODE}"

case "$MODE" in
  node)
    MODE_ARGS=(--cuda-graph-trace=node)
    ;;
  pytrace_graph)
    MODE_ARGS=(--cuda-graph-trace=graph --pytorch=functions-trace)
    ;;
  *)
    echo "Invalid mode: $MODE. Expected 'node' or 'pytrace_graph'." >&2
    exit 2
    ;;
esac

uv run --with nvtx nsys profile \
  --session-new="$SESSION" \
  --start-later=true \
  --trace=cuda,nvtx \
  "${MODE_ARGS[@]}" \
  --sample=none \
  --cpuctxsw=none \
  --output="$REPORT" \
  --export=sqlite \
  --force-overwrite=true \
  --wait=all \
  bash "$VLLM_SCRIPT"