"""P21 (P18-B): OpenAI-compatible endpoint in front of the PD control plane.

Spawns the PD pair (D card1 + P card0), then serves a minimal
POST /v1/chat/completions (non-streaming) + GET /health on card0's spare
capacity: chat-template → framed request to the P router → result back as
OpenAI JSON.  `priority` in the body rides through to the D-side scheduler
(P18-C).

Run:  PYTHONPATH=. python3 scripts/p21_pd_openai.py \
        --model /root/autodl-tmp/models/Qwen3-0.6B --port 8100
Smoke: curl -s -X POST localhost:8100/v1/chat/completions -H 'Content-Type:
       application/json' -d '{"messages":[{"role":"user","content":"hi"}]}'
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import uuid

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from nanovllm.pd.protocol import JsonStream, connect_retry, send_json  # noqa: E402


class PDClient:
    """Thread-safe framed client for the P router (one connection, lock)."""

    def __init__(self, host, port):
        self.lock = threading.Lock()
        self.sock = connect_retry(host, port)
        self.sock.setblocking(True)

    def complete(self, prompt, max_tokens, priority):
        with self.lock:
            rid = uuid.uuid4().hex
            send_json(self.sock, {"req_id": rid, "prompt": prompt,
                                  "max_tokens": max_tokens,
                                  "priority": priority})
            js = JsonStream()
            while True:
                chunk = self.sock.recv(1 << 20)
                if not chunk:
                    raise ConnectionError("P router closed")
                for msg in js.feed(chunk):
                    if msg.get("type") == "result" and msg["req_id"] == rid:
                        return msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    parser.add_argument("--gpu-util", type=float, default=0.4)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--p-port", type=int, default=29750)
    parser.add_argument("--d-port", type=int, default=29760)
    args = parser.parse_args()

    env = dict(os.environ, PYTHONPATH=REPO)
    log_dir = "/root/p21_data"
    os.makedirs(log_dir, exist_ok=True)
    procs = [subprocess.Popen(
        [sys.executable, os.path.join(REPO, "scripts/p15_pd_serve.py"),
         "--role", "decode", "--model", args.model,
         "--gpu-util", str(args.gpu_util), "--port", str(args.d_port)],
        env={**env, "MUSA_VISIBLE_DEVICES": "1"},
        stdout=open(f"{log_dir}/openai_decode.log", "w"), stderr=subprocess.STDOUT)]
    procs.append(subprocess.Popen(
        [sys.executable, os.path.join(REPO, "scripts/p15_pd_serve.py"),
         "--role", "prefill", "--model", args.model,
         "--gpu-util", str(args.gpu_util), "--port", str(args.p_port),
         "--d-port", str(args.d_port)],
        env=env, stdout=open(f"{log_dir}/openai_prefill.log", "w"),
        stderr=subprocess.STDOUT))
    pd = PDClient("127.0.0.1", args.p_port)

    from fastapi import FastAPI, HTTPException, Request
    import uvicorn
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": "pd", "model": args.model}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(400, "messages must be a non-empty list")
        try:
            priority = int(body.get("priority", 0))
        except (TypeError, ValueError):
            raise HTTPException(400, "priority must be an integer")
        max_tokens = int(body.get("max_tokens", 64))
        prompt_ids = tok.apply_chat_template(messages, tokenize=True,
                                             add_generation_prompt=True)
        if hasattr(prompt_ids, "keys"):
            prompt_ids = prompt_ids["input_ids"]
        if hasattr(prompt_ids, "tolist"):
            prompt_ids = prompt_ids.tolist()
        prompt = prompt_ids if isinstance(prompt_ids, list) else prompt_ids[0]
        t0 = time.perf_counter()
        msg = pd.complete(prompt, max_tokens, priority)
        text = tok.decode(msg["tokens"])
        return {
            "id": f"chatcmpl-{msg['req_id'][:12]}",
            "object": "chat.completion",
            "model": args.model,
            "choices": [{"index": 0, "message": {"role": "assistant",
                                                 "content": text},
                         "finish_reason": "length"}],
            "usage": {"prompt_tokens": len(prompt),
                      "completion_tokens": len(msg["tokens"])},
            "pd_metrics": {"prefill_s": msg.get("prefill_s"),
                           "ttft_s": round(time.perf_counter() - t0, 3)},
        }

    try:
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
    finally:
        for proc in procs:
            proc.terminate()


if __name__ == "__main__":
    main()
