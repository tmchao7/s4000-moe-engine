#!/bin/bash
# P14-B1: does MCCL_GRAPH_MIXING_SUPPORT change the 0.6B TP2 paged whole-graph first-replay wedge (I29)?
# Usage (server): setsid nohup bash scripts/p14_mccl_probe.sh > /root/p14_data/mccl_probe.log 2>&1 &
# Arms (interleaved x3): base (env unset) | mix1 (=1) | mix0 (=0 explicit).
# Each graph run is hard-bounded by timeout; a timeout == WEDGE. Cleanup is by PID after each run.
set -u
cd /root/moe-server-s4000
OUT=/root/p14_data/mccl; mkdir -p "$OUT"
MODEL=/root/autodl-tmp/models/Qwen3-0.6B
LOCK=/tmp/nanovllm-p14.lock
PAT='^/usr/bin/python3|^python3|^[s]pawn_main'
if [ -e "$LOCK" ]; then echo "lock exists"; exit 2; fi
if pgrep -f "$PAT" >/dev/null; then echo "foreign processes present -> abort"; pgrep -fa "$PAT"; exit 3; fi
touch "$LOCK"; trap 'rm -f "$LOCK"' EXIT
COMMON="--tp 2 --model $MODEL --dense --attention-backend triton_paged --max-model-len 1024 --gpu-mem 0.3 --generates 5 --max-tokens 8"

clean() {
  local pids; pids=$(pgrep -f "$PAT" | tr '\n' ' ')
  if [ -n "$pids" ]; then echo "[clean] killing PIDs: $pids"; kill -9 $pids 2>/dev/null; sleep 5; fi
  for f in /dev/shm/nanovllm_*; do [ -e "$f" ] || continue; if fuser "$f" >/dev/null 2>&1; then echo "[clean] shm $f still held"; else rm -f "$f"; echo "[clean] removed $f"; fi; done
  for i in $(seq 1 24); do
    used=$(mthreads-gmi | grep -o '[0-9]*MiB(' | tr -d 'MiB(' | sort -nr | head -1)
    [ "${used:-0}" -le 10 ] && break; sleep 5
  done
  echo "[clean] gpu max used MiB=${used:-?}"
}

echo "[p14-probe] start $(date -Is)"; mthreads-gmi | grep MiB
echo "[p14-probe] eager reference"
PYTHONPATH=. timeout -k 5 600 python3 tests/p8_graph_padfix.py $COMMON --phase eager --output $OUT/eager06.json > $OUT/eager06.log 2>&1
echo "[p14-probe] eager rc=$?"; clean

for rep in 1 2 3; do
  for arm in base mix1 mix0; do
    tag="${arm}_r${rep}"
    case $arm in base) E="";; mix1) E="MCCL_GRAPH_MIXING_SUPPORT=1";; mix0) E="MCCL_GRAPH_MIXING_SUPPORT=0";; esac
    echo "[p14-probe] === $tag env='$E' $(date -Is)"
    t0=$(date +%s)
    env $E PYTHONPATH=. NANOVLLM_TRACE_INIT=0 timeout -k 5 420 python3 tests/p8_graph_padfix.py $COMMON --phase graph --eager-from $OUT/eager06.json --output $OUT/$tag.json > $OUT/$tag.log 2>&1
    rc=$?; dt=$(( $(date +%s) - t0 ))
    echo "[p14-probe] $tag rc=$rc dt=${dt}s generates_done=$(grep -c 'graph generate' $OUT/$tag.log)"
    echo "{\"tag\":\"$tag\",\"arm\":\"$arm\",\"env\":\"$E\",\"rc\":$rc,\"seconds\":$dt}" > $OUT/$tag.meta.json
    clean
  done
done
echo "[p14-probe] done $(date -Is)"
