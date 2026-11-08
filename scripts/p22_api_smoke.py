"""P22-A API smoke: the new serving surface against a live api_server.

Spawns api_server (TP2 default CLI), then exercises:
  1  chat knobs (top_p/top_k/frequency_penalty) -> 200 + content
  2  chat logprobs=true + top_logprobs=3 -> choices[0].logprobs.content
  3  chat top_p=0 -> 400 (SamplingParams validation mapped to 400)
  4  completions non-stream + logprobs=3 -> legacy logprobs arrays
  5  completions stream=true -> SSE text_completion.chunk deltas
  6  chat stop_token_ids accepted

Run:  PYTHONPATH=. python3 scripts/p22_api_smoke.py [model]
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8127


def post(path, body, stream=False, timeout=180):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=timeout)
    if stream:
        return resp
    return json.loads(resp.read())


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/models/Qwen3-0.6B"
    proc = subprocess.Popen(
        [sys.executable, "nanovllm/entrypoints/openai/api_server.py",
         "--model", model, "--port", str(PORT), "--max-model-len", "2048"],
        env={**os.environ, "PYTHONPATH": REPO},
        stdout=open("/root/p22_api_server.log", "w"), stderr=subprocess.STDOUT)
    ok = True
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("server never became healthy; see /root/p22_api_server.log")

        r = post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "用一句话介绍 KV cache。"}],
            "max_tokens": 32, "temperature": 0.8, "top_p": 0.9, "top_k": 50,
            "frequency_penalty": 0.2})
        good = (r["choices"][0]["finish_reason"] in ("stop", "length")
                and len(r["choices"][0]["message"]["content"]) > 0)
        print("1 chat knobs:", "OK" if good else "FAIL")
        ok &= good

        r = post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "数到三。"}],
            "max_tokens": 8, "temperature": 0, "logprobs": True, "top_logprobs": 3})
        lp = r["choices"][0]["logprobs"]
        good = bool(lp and lp["content"] and lp["content"][0]["top_logprobs"]
                    and len(lp["content"][0]["top_logprobs"]) == 3)
        print("2 chat logprobs:", "OK" if good else "FAIL",
              json.dumps(lp["content"][0], ensure_ascii=False)[:200] if lp else "")
        ok &= good

        try:
            post("/v1/chat/completions", {
                "messages": [{"role": "user", "content": "x"}], "top_p": 0})
            print("3 validation 400: FAIL (no error raised)")
            ok = False
        except urllib.error.HTTPError as exc:
            print(f"3 validation 400: {'OK' if exc.code == 400 else 'FAIL'} ({exc.code})")
            ok &= exc.code == 400

        r = post("/v1/completions", {
            "prompt": "大模型推理中的 KV cache 是", "max_tokens": 24,
            "temperature": 0.7, "top_p": 0.9, "logprobs": 3})
        ch = r["choices"][0]
        good = (len(ch["text"]) > 0 and ch["logprobs"]
                and len(ch["logprobs"]["tokens"]) == r["usage"]["completion_tokens"]
                and len(ch["logprobs"]["top_logprobs"][0]) <= 3)
        print("4 completions + logprobs:", "OK" if good else "FAIL")
        ok &= good

        resp = post("/v1/completions", {
            "prompt": "写一首两句的诗：", "max_tokens": 16,
            "temperature": 0.7, "stream": True}, stream=True)
        chunks, text = 0, ""
        for raw in resp:
            line = raw.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                data = json.loads(line[6:])
                if data.get("choices") and data["choices"]:
                    text += data["choices"][0]["text"]
                    chunks += 1
        good = chunks > 1 and len(text) > 0
        print(f"5 completions SSE: {'OK' if good else 'FAIL'} ({chunks} chunks)")
        ok &= good

        r = post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "你好"}],
            "max_tokens": 16, "temperature": 0, "stop_token_ids": [151643]})
        good = r["choices"][0]["finish_reason"] in ("stop", "length")
        print("6 chat stop_token_ids:", "OK" if good else "FAIL")
        ok &= good

        print("P22 API SMOKE", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        proc.terminate()
        proc.wait(timeout=120)


if __name__ == "__main__":
    sys.exit(main())
