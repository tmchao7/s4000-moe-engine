#!/bin/bash
# Sync the working tree to the MUSA box and run the full 15-suite regression
# there.  Strong gate for the P26+ audit workflow (and usable by hand).
# Requires the ssh alias `s4000` (see docs/SETUP.md); ~35 min on the box.
set -eu
cd "$(dirname "$0")/.."
tar czf - nanovllm tests scripts docs 2>/dev/null | ssh -o ConnectTimeout=15 s4000 'tar xzf - -C /root/moe-server-s4000'
ssh -o ConnectTimeout=15 s4000 'cd /root/moe-server-s4000 && bash tests/run_regression.sh --model /root/autodl-tmp/models/Qwen3-0.6B'
