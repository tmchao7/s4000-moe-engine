#!/bin/bash
# Smoke-test the two-replica DP2 launcher (health, models, concurrent chat).
set -euo pipefail
PORTS_STR=$(printenv S4000_DP2_PORTS || echo "8100 8101")
SERVED_NAME=$(printenv S4000_DP2_SERVED_NAME || echo s4000-dp2)
API_KEY=$(printenv S4000_DP2_API_KEY || true)
export PORTS_STR SERVED_NAME API_KEY
python3 - <<'PY'
import concurrent.futures
import json
import os
import urllib.request

ports = os.environ["PORTS_STR"].split()
model = os.environ["SERVED_NAME"]
api_key = os.environ.get("API_KEY", "")

def req(port, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                               headers=headers, method="GET" if body is None else "POST")
    with urllib.request.urlopen(r, timeout=120) as resp:
        return json.loads(resp.read().decode())

def one(port):
    health = req(port, "/health")
    models = req(port, "/v1/models")
    chat = req(port, "/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 8,
        "temperature": 0,
    })
    choice = chat["choices"][0]["message"]["content"]
    return {"port": port, "health": health.get("status"),
            "models": [m["id"] for m in models.get("data", [])],
            "choice": choice}

with concurrent.futures.ThreadPoolExecutor(max_workers=len(ports)) as ex:
    results = list(ex.map(one, ports))
for r in results:
    print(r)
    assert r["health"] == "ok", r
    assert model in r["models"], r
    assert r["choice"], r
print("[dp2-smoke] PASS")
PY
