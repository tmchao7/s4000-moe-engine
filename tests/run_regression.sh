#!/bin/bash
# One-command regression: the "don't rely on remembering" gate (P26/B7).
#
# Runs every live suite in fast-fail order and exits rc=1 on any failure.
# Any change to scheduler / state-machine / runtime-switch code MUST go
# through this script before commit — the I37/I38 lesson: judged gates that
# only run some suites let same-family bugs ship twice.
#
# Usage:  bash tests/run_regression.sh [--model /path/to/Qwen3-0.6B]
#         (run from anywhere; cd's to the repo root; ~25 min on a MUSA box)
# Suites without --model in their argparse use their built-in default.
set -u
cd "$(dirname "$0")/.."
MODEL_FLAG=""
if [ "${1:-}" = "--model" ]; then MODEL_FLAG="--model $2"; fi
OUT_DIR=${REGRESSION_OUT:-/tmp/regression}
mkdir -p "$OUT_DIR"
rc_all=0

run() {
  local tag="$1"; shift
  echo "===== $tag start $(date +%H:%M:%S) ====="
  "$@"; local rc=$?
  echo "===== $tag rc=$rc ====="
  if [ $rc -ne 0 ]; then rc_all=1; fi
}

# CPU-only first (fast fail), then GPU suites tiny→feature.
run repro   env PYTHONPATH=. python3 tests/p25_review_repro.py
run pytest  python3 -m pytest tests/test_benchmarking_v3.py -q
run p8      env PYTHONPATH=. python3 tests/p8_scheduler_gaps.py
run p2      env PYTHONPATH=. python3 tests/p2_grouped_mm.py
run p6      env PYTHONPATH=. python3 tests/p6_bmm_dense_tiny.py
run p23     env PYTHONPATH=. python3 tests/p23_overlap.py $MODEL_FLAG --out "$OUT_DIR/p23.json"
run p25fix  env PYTHONPATH=. python3 tests/p25_overlap_fix.py $MODEL_FLAG --out "$OUT_DIR/p25_overlap_fix.json"

# Feature regression nets (judged capabilities; keep green or justify).
run p16lru  env PYTHONPATH=. python3 tests/p16_lru_evict.py
run p16mix  env PYTHONPATH=. python3 tests/p16_mixed_unified.py $MODEL_FLAG --out "$OUT_DIR/p16_mixed.json"
run p16pri  env PYTHONPATH=. python3 tests/p16_priority_sched.py
run p17unit env PYTHONPATH=. python3 tests/p17_kv_int8.py --mode unit
run p22sam  env PYTHONPATH=. python3 tests/p22_sampler_face.py $MODEL_FLAG
run p22fix  env PYTHONPATH=. python3 tests/p22_bugfix_regress.py $MODEL_FLAG
run p24     env PYTHONPATH=. python3 tests/p24_guided.py $MODEL_FLAG --out "$OUT_DIR/p24.json"

echo "[regression] rc_all=$rc_all end $(date)"
exit $rc_all
