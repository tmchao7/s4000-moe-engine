#!/bin/bash
# Sync the working tree to the MUSA box and run the full 16-suite regression
# there.  Strong gate for the P26+ audit workflow (and usable by hand).
# Requires the ssh alias `s4000` (see docs/SETUP.md) and rsync on the box;
# ~35 min on the box.
set -euo pipefail
cd "$(dirname "$0")/.."
# Mirror each synced dir with rsync --delete-after.  A tar-over-ssh overlay
# keeps stale files (e.g. scripts moved to scripts/archive), which made
# pytest pass on the server but fail from a clean checkout; and a tar pipe
# is brittle on a flaky ssh link (Write error).  Per-dir rsync keeps
# unrelated root files (e.g. the fake .git) alone and deletes stale files
# only after a file has transferred.  A dropped connection leaves the old
# tree in place for the next retry.
for d in nanovllm tests scripts docs; do
  rsync -a --delete-after --exclude='__pycache__/' --exclude='.DS_Store' \
    "$d/" "s4000:/root/moe-server-s4000/$d/"
done
ssh -o ConnectTimeout=15 s4000 \
  'test -f /root/moe-server-s4000/nanovllm/engine/scheduler.py && test -f /root/moe-server-s4000/tests/run_regression.sh'
ssh -o ConnectTimeout=15 s4000 'cd /root/moe-server-s4000 && bash tests/run_regression.sh --model /root/autodl-tmp/models/Qwen3-0.6B'
