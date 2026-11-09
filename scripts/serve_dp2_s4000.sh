#!/bin/bash
# Start/stop two independent S4000 server replicas (one GPU each).
#
# This is the "multi-replica operations" path demonstrated with 0.6B on 2x S4000.
# It is NOT a 30B DP claim: a 30B replica needs FP8 retention, which the current
# loader does not implement. Keep the model small (e.g. Qwen3-0.6B) unless the
# single-card memory budget is verified.
#
# Usage:
#   S4000_DP2_MODEL=/root/autodl-tmp/models/Qwen3-0.6B \
#     S4000_DP2_PORTS="8100 8101" S4000_DP2_GPUS="0 1" \
#     scripts/serve_dp2_s4000.sh start
#   scripts/serve_dp2_s4000.sh status
#   scripts/serve_dp2_s4000.sh stop
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_DIR=$(printenv S4000_DP2_RUN_DIR || echo /tmp/s4000-dp2)
MODEL=$(printenv S4000_DP2_MODEL || true)
PORTS_STR=$(printenv S4000_DP2_PORTS || echo "8100 8101")
GPUS_STR=$(printenv S4000_DP2_GPUS || echo "0 1")
MAX_LEN=$(printenv S4000_DP2_MAX_MODEL_LEN || echo 4096)
SERVED_NAME=$(printenv S4000_DP2_SERVED_NAME || echo s4000-dp2)
GPU_UTIL=$(printenv S4000_DP2_GPU_UTIL || echo 0.6)
HEALTH_TIMEOUT=$(printenv S4000_DP2_HEALTH_TIMEOUT || echo 1800)
API_KEY=$(printenv S4000_DP2_API_KEY || true)
PYTHON=$(printenv S4000_PYTHON || echo python3)
read -r PORT0 PORT1 <<< "$PORTS_STR"
read -r GPU0 GPU1 <<< "$GPUS_STR"

usage() {
  echo "usage: $0 {start|stop|status}" >&2
  exit 2
}

check_cfg() {
  [ -n "$PORT0" ] && [ -n "$PORT1" ] || { echo "need two ports" >&2; exit 2; }
  [ -n "$GPU0" ] && [ -n "$GPU1" ] || { echo "need two GPU ids" >&2; exit 2; }
  [ -n "$MODEL" ] || { echo "S4000_DP2_MODEL is required" >&2; exit 2; }
  [ -d "$MODEL" ] || { echo "model dir not found: $MODEL" >&2; exit 2; }
}

pid_file() { echo "$RUN_DIR/replica-$1.pid"; }
log_file() { echo "$RUN_DIR/replica-$1.log"; }
port_file() { echo "$RUN_DIR/replica-$1.port"; }

start_one() {
  local rank=$1 gpu=$2 port=$3
  local pidf logf
  pidf=$(pid_file "$rank")
  logf=$(log_file "$rank")
  if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "replica $rank already running (pid $(cat "$pidf"))" >&2
    return 1
  fi
  if [ -n "$API_KEY" ]; then
    MUSA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" \
      nohup "$PYTHON" "$ROOT/nanovllm/entrypoints/openai/api_server.py" \
      --model "$MODEL" --host 127.0.0.1 --port "$port" \
      --served-model-name "$SERVED_NAME" \
      --tensor-parallel-size 1 --expert-parallel-size 1 \
      --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL" \
      --api-key "$API_KEY" >"$logf" 2>&1 &
  else
    MUSA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" \
      nohup "$PYTHON" "$ROOT/nanovllm/entrypoints/openai/api_server.py" \
      --model "$MODEL" --host 127.0.0.1 --port "$port" \
      --served-model-name "$SERVED_NAME" \
      --tensor-parallel-size 1 --expert-parallel-size 1 \
      --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL" \
      >"$logf" 2>&1 &
  fi
  echo $! >"$pidf"
  echo "$port" >"$(port_file "$rank")"
  echo "started replica $rank gpu=$gpu port=$port pid=$(cat "$pidf") log=$logf"
}

wait_health() {
  local rank=$1 port=$2 pidf deadline
  pidf=$(pid_file "$rank")
  deadline=$((SECONDS + HEALTH_TIMEOUT))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if ! kill -0 "$(cat "$pidf")" 2>/dev/null; then
      echo "replica $rank died while loading; tail $(log_file "$rank")" >&2
      tail -40 "$(log_file "$rank")" >&2 || true
      return 1
    fi
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "replica $rank healthy on port $port"
      return 0
    fi
    sleep 2
  done
  echo "replica $rank health timeout; tail $(log_file "$rank")" >&2
  tail -40 "$(log_file "$rank")" >&2 || true
  return 1
}

cmd_start() {
  check_cfg
  mkdir -p "$RUN_DIR"
  start_one 0 "$GPU0" "$PORT0"
  start_one 1 "$GPU1" "$PORT1"
  wait_health 0 "$PORT0"
  wait_health 1 "$PORT1"
  echo "DP2 ready: ports $PORT0 $PORT1, served_name=$SERVED_NAME"
}

cmd_stop() {
  local rank pidf pid
  for rank in 0 1; do
    pidf=$(pid_file "$rank")
    if [ -f "$pidf" ]; then
      pid=$(cat "$pidf")
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        for _ in $(seq 1 30); do
          kill -0 "$pid" 2>/dev/null || break
          sleep 1
        done
        if kill -0 "$pid" 2>/dev/null; then
          kill -9 "$pid" 2>/dev/null || true
        fi
      fi
      rm -f "$pidf" "$(port_file "$rank")"
    fi
  done
  echo "DP2 stopped"
}

cmd_status() {
  local rank pidf pid port
  for rank in 0 1; do
    pidf=$(pid_file "$rank")
    if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
      pid=$(cat "$pidf")
      port=$(cat "$(port_file "$rank")" 2>/dev/null || echo "?")
      if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
        echo "replica $rank pid=$pid port=$port HEALTHY"
      else
        echo "replica $rank pid=$pid port=$port NOT_HEALTHY"
      fi
    else
      echo "replica $rank STOPPED"
    fi
  done
}

case "$1" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  *) usage ;;
esac
