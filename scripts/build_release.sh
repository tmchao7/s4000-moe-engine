#!/bin/bash
# Build a slim deployment tarball from a committed tree.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)
OUT="dist/s4000-moe-engine-${VERSION}.tar.gz"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "[release] working tree is dirty; commit first" >&2
  exit 1
fi
mkdir -p dist
rm -f "$OUT"
git archive --format=tar.gz --prefix="s4000-moe-engine-${VERSION}/" -o "$OUT" HEAD -- \
  nanovllm pyproject.toml README.md LICENSE NOTICE requirements-serve.txt \
  deploy docs/DEPLOY.md docs/OPERATIONS.md docs/SETUP.md docs/musa-pitfalls.md \
  scripts/install_s4000.sh scripts/serve_s4000.sh scripts/smoke_serve.sh \
  scripts/build_release.sh

LIST=$(tar -tzf "$OUT")
for forbidden in benchmarks/ tests/ assets/ scripts/archive/; do
  if printf '%s\n' "$LIST" | grep -q "/$forbidden"; then
    echo "[release] forbidden path in archive: $forbidden" >&2
    exit 1
  fi
done
for required in "nanovllm/entrypoints/openai/serve.py" "deploy/serve.env.example" "scripts/serve_s4000.sh" "docs/DEPLOY.md"; do
  if ! printf '%s\n' "$LIST" | grep -q "/$required"; then
    echo "[release] missing $required" >&2
    exit 1
  fi
done

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
tar xzf "$OUT" -C "$TMP"
python3 -m compileall -q "$TMP/s4000-moe-engine-${VERSION}/nanovllm"
echo "[release] built $OUT"
echo "[release] contents verified (no benchmarks/tests/assets; required files present)"