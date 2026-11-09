#!/bin/bash
# Product smoke test against an already-running s4000-moe-engine service.
set -euo pipefail
BASE=http://127.0.0.1:8000
API_KEY=${NANOVLLM_API_KEY:-}

while [ $# -gt 0 ]; do
  case "$1" in
    --base-url) BASE=$2; shift 2 ;;
    --api-key) API_KEY=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

AUTH=()
if [ -n "$API_KEY" ]; then AUTH=(-H "Authorization: Bearer $API_KEY"); fi
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

echo "[smoke] health $BASE/health"
ok=0
for _ in $(seq 1 60); do
  if curl -fsS "$BASE/health" -o "$TMP" 2>/dev/null; then ok=1; break; fi
  sleep 1
done
if [ "$ok" -ne 1 ]; then echo "[smoke] health failed" >&2; exit 1; fi
python3 - "$TMP" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p.get("status") == "ok", p
assert p.get("model"), p
print("[smoke] health OK model=", p["model"], "version=", p.get("version"))
PY

curl -fsS "${AUTH[@]}" "$BASE/v1/models" -o "$TMP"
python3 - "$TMP" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p.get("object") == "list" and p.get("data"), p
print("[smoke] models OK", p["data"][0].get("id"))
PY

curl -fsS "${AUTH[@]}" "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello in one short sentence."}],"max_tokens":16,"temperature":0}' -o "$TMP"
python3 - "$TMP" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert "metrics" not in p, "non-stream response must not contain metrics"
assert p.get("choices") and p["choices"][0].get("message", {}).get("content"), p
assert p.get("usage", {}).get("prompt_tokens"), p
print("[smoke] chat OK", p["choices"][0]["message"]["content"][:80])
PY

curl -fsSN "${AUTH[@]}" "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Count to three."}],"max_tokens":16,"temperature":0,"stream":true}' -o "$TMP"
grep -q 'data: \[DONE\]' "$TMP"
grep -q 'chat.completion.chunk' "$TMP"
echo "[smoke] stream OK"
echo "[smoke] PASS"