"""CPU-only regression for ``scripts/benchmark_online.py``.

The tiny MoE checkpoint ships without a chat template, so the HTTP benchmark
client is validated against a stdlib SSE stub instead of a real engine.  This
keeps the client's parsing, timing, percentile and ``/metrics`` scraping logic
honest before it is pointed at the 30B server, where a bad run costs a full
model load.

    python3 tests/p4_benchmark_client.py
"""
import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNKS = 4
TOKEN_DELAY = 0.02


class StubState:

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = 0
        self.mixed = 0
        self.omit_done = False

    def record_request(self):
        with self.lock:
            self.requests += 1
            self.mixed = self.requests // 2
            return self.requests, self.mixed

    def snapshot(self):
        with self.lock:
            return self.requests, self.mixed


def build_handler(state):

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _read_body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length) or b"{}")

        def _plain(self, text):
            payload = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _chunk(self, text):
            payload = text.encode("utf-8")
            self.wfile.write(f"{len(payload):X}\r\n".encode("ascii") + payload + b"\r\n")

        def do_GET(self):
            if self.path != "/metrics":
                self.send_error(404)
                return
            requests, mixed = state.snapshot()
            self._plain(
                f"nanovllm_requests_total {requests}\n"
                f"nanovllm_engine_mixed_steps_total {mixed}\n"
                f"nanovllm_engine_decode_steps_total {mixed * 2}\n"
            )

        def do_POST(self):
            self._read_body()
            state.record_request()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for _ in range(CHUNKS):
                payload = json.dumps({
                    "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}],
                })
                self._chunk(f"data: {payload}\n\n")
                time.sleep(TOKEN_DELAY)
            usage = json.dumps({
                "choices": [],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": CHUNKS,
                    "total_tokens": 12 + CHUNKS,
                },
            })
            self._chunk(f"data: {usage}\n\n")
            if not state.omit_done:
                self._chunk("data: [DONE]\n\n")
            self._chunk("")

    return Handler


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    state = StubState()
    port = free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), build_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = Path(tempfile.mkdtemp(prefix="p4-online-")) / "online.json"
    try:
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "benchmark_online.py"),
                "--base-url", f"http://127.0.0.1:{port}",
                "--model", "stub",
                "--max-qps", "20",
                "--requests", "8",
                "--warmup-requests", "1",
                "--max-tokens", "4",
                "--fractions", "0.5,1.0",
                "--arrival-pattern", "fixed",
                "--output", str(output),
            ],
            check=True,
            cwd=REPO_ROOT,
        )
    finally:
        server.shutdown()
        server.server_close()

    payload = json.loads(output.read_text(encoding="utf-8"))
    summaries = payload["summary"]
    assert len(summaries) == 2, summaries
    assert len(payload["records"]) == 16, len(payload["records"])
    assert len(payload["points"]) == 2, payload["points"]
    assert payload["schema_version"] == 2
    for row in summaries:
        assert row["success_rate"] == 1.0, row
        assert row["achieved_qps"] > 0, row
        assert row["ttft_p50_s"] and row["ttft_p50_s"] > 0, row
        assert row["tpot_p95_s"] and row["tpot_p95_s"] > 0, row
        assert row["output_tokens_per_s"] > 0, row
        assert row["max_concurrency"] > 0, row
        assert row["mixed_steps"] > 0, row
    assert all(row["done_received"] for row in payload["records"])
    assert all(row["output_tokens"] == CHUNKS for row in payload["records"])
    missing_done_output = output.with_name("missing-done.json")
    missing_state = StubState()
    missing_state.omit_done = True
    missing_port = free_port()
    missing_server = ThreadingHTTPServer(
        ("127.0.0.1", missing_port), build_handler(missing_state)
    )
    missing_thread = threading.Thread(target=missing_server.serve_forever, daemon=True)
    missing_thread.start()
    try:
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "benchmark_online.py"),
                "--base-url", f"http://127.0.0.1:{missing_port}",
                "--model", "stub",
                "--max-qps", "20",
                "--requests", "2",
                "--warmup-requests", "0",
                "--max-tokens", "4",
                "--fractions", "1.0",
                "--arrival-pattern", "fixed",
                "--output", str(missing_done_output),
            ],
            check=True,
            cwd=REPO_ROOT,
        )
    finally:
        missing_server.shutdown()
        missing_server.server_close()
    missing_done = json.loads(missing_done_output.read_text(encoding="utf-8"))
    assert missing_done["summary"][0]["success_rate"] == 0.0
    assert all("before [DONE]" in row["error"] for row in missing_done["records"])
    assert output.with_suffix(".md").is_file()
    assert output.with_suffix(".jsonl").is_file()
    print("[online-client] stub SSE benchmark passed", summaries)


if __name__ == "__main__":
    main()
