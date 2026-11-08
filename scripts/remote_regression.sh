#!/bin/bash
# Sync the working tree to the MUSA box and run the full 16-suite regression
# there.  Strong gate for the P26+ audit workflow (and usable by hand).
# Requires the ssh alias `s4000` (see docs/SETUP.md); ~35 min on the box.
set -eu
cd "$(dirname "$0")/.."
# Mirror the tree instead of overlaying it: a plain tar extraction keeps
# stale files from older runs (e.g. scripts that were since moved to
# scripts/archive), which made pytest pass on the server but fail from a
# clean checkout.  Remove the four synced dirs first so the gate sees exactly
# the checked-in tree.  The target path is fixed and verified.
tar czf - nanovllm tests scripts docs 2>/dev/null | \
  ssh -o ConnectTimeout=15 s4000 \
  'cd /root/moe-server-s4000 && rm -rf nanovllm tests scripts docs && tar xzf -'
ssh -o ConnectTimeout=15 s4000 'cd /root/moe-server-s4000 && bash tests/run_regression.sh --model /root/autodl-tmp/models/Qwen3-0.6B'
