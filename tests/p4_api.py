"""Schema and streaming smoke test for the OpenAI-compatible API."""
from queue import Queue
import sys

sys.path.insert(0, "/root/moe-server-s4000")

from nanovllm.async_engine import RequestHandle, TokenEvent
from nanovllm.entrypoints.openai.api_server import create_app


class FakeTokenizer:

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        return [1, 2, 3]

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


class FakeBatchEncodingTokenizer(FakeTokenizer):
    """transformers 5.x returns a BatchEncoding instead of a plain id list."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        super().apply_chat_template(messages, tokenize, add_generation_prompt)
        return {"input_ids": [1, 2, 3]}


class FakeEngine:

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or FakeTokenizer()
        self.cancelled = []

    async def add_request(self, prompt, sampling_params):
        events = Queue()
        events.put(TokenEvent(token_id=65))
        events.put(TokenEvent(token_id=66, finished=True, metrics={"ttft_s": 0.01}))
        return RequestHandle(7, events)

    async def next_event(self, handle):
        return handle.events.get()

    def cancel(self, seq_id):
        self.cancelled.append(seq_id)

    async def close(self):
        return None

    def metrics(self):
        return {"requests_total": 2, "active_requests": 0}


class FakeOverloadedEngine(FakeEngine):

    async def add_request(self, prompt, sampling_params):
        raise OverflowError("engine request queue is full")


def main():
    from fastapi.testclient import TestClient

    request = {
        "model": "tiny-moe",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
        "max_tokens": 2,
    }
    for tokenizer in (FakeTokenizer(), FakeBatchEncodingTokenizer()):
        app = create_app(FakeEngine(tokenizer), "tiny-moe")
        with TestClient(app) as client:
            assert client.get("/health").json()["status"] == "ok"
            assert client.get("/v1/models").json()["data"][0]["id"] == "tiny-moe"
            response = client.post("/v1/chat/completions", json=request)
            assert response.status_code == 200, response.text
            assert response.json()["choices"][0]["message"]["content"] == "AB"
            assert response.json()["usage"]["completion_tokens"] == 2
            assert response.json()["usage"]["prompt_tokens"] == 3
            with client.stream(
                "POST", "/v1/chat/completions", json={
                    **request,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
            ) as response:
                body = "".join(response.iter_text())
            assert "chat.completion.chunk" in body
            assert '"completion_tokens": 2' in body
            assert "data: [DONE]" in body
            metrics = client.get("/metrics").text
            assert "# TYPE nanovllm_requests_total counter" in metrics
            assert "nanovllm_requests_total 2" in metrics
        print(f"[api] {type(tokenizer).__name__}: stream+nonstream/metrics passed")
    overloaded = create_app(FakeOverloadedEngine(), "tiny-moe")
    with TestClient(overloaded) as client:
        response = client.post("/v1/chat/completions", json=request)
        assert response.status_code == 429, response.text
    print("[api] health/models/chat stream+nonstream/metrics passed")


if __name__ == "__main__":
    main()
