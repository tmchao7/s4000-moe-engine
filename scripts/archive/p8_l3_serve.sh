#!/usr/bin/env bash
# L3 server launcher (P4.3 long-context retest).
# Usage: p8_l3_serve.sh <chunk-size> <log-file>
set -u
CHUNK="${1:-256}"
LOG="${2:-/root/p8_l3_server_${CHUNK}.log}"
MODEL=/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8

pgrep -fa "^/usr/bin/python3|^python3|^[s]pawn_main" && {
  echo "[l3] GPU busy, refusing to start"; exit 1;
}
rm -f /dev/shm/nanovllm*
cd /root/moe-server-s4000
NANOVLLM_TRACE_INIT=1 PYTHONPATH=/root/moe-server-s4000 setsid nohup python3 nanovllm/entrypoints/openai/api_server.py \
  --model "$MODEL" \
  --served-model-name qwen3-30b-a3b \
  --attention-backend triton_paged \
  --moe-backend grouped_mm \
  --decode-moe-backend bmm_dense \
  --scheduler-policy stall_free \
  --prefill-chunk-size "$CHUNK" \
  --max-model-len 4096 \
  --metrics-path "/root/p8_l3_metrics_${CHUNK}.jsonl" \
  > "$LOG" 2>&1 &
echo "[l3] server pid $! chunk=$CHUNK log=$LOG"
