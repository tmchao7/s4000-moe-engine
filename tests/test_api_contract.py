"""OpenAI API product contract tests with a fake engine (CPU; no GPU)."""
import asyncio

import pytest

pytest.importorskip("httpx")
pytest.importorskip("fastapi")
pytest.importorskip("tqdm")
pytest.importorskip("torch")

from fastapi.testclient import TestClient

from nanovllm.async_engine import TokenEvent
from nanovllm.entrypoints.openai.api_server import create_app


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [1, 2, 3]

    def encode(self, text):
        return [1, 2, 3]

    def decode(self, token_ids, skip_special_tokens=True):
        return "hello"


class FakeEngine:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.sent = False

    async def add_request(self, prompt, sampling_params):
        return type("Handle", (), {"seq_id": 1})()

    async def next_event(self, handle):
        if not self.sent:
            self.sent = True
            return TokenEvent(token_id=42, metrics={
                "finish_reason": "stop", "prompt_tokens": 3,
                "output_tokens": 1, "ttft_s": 0.01, "e2e_s": 0.02,
                "tpot_s": 0.01, "mean_tpot_s": 0.01, "itls_s": [],
                "queue_s": 0.0,
            })
        return TokenEvent(finished=True)

    def cancel(self, seq_id):
        return None

    async def close(self):
        return None

    def metrics(self):
        return {"requests_total": 1}


def _client(api_key=None):
    engine = FakeEngine()
    app = create_app(engine, "fake-model", api_key=api_key, version="test")
    return TestClient(app)


def test_health_and_models():
    with _client() as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert r.json()["version"] == "test"
        r = client.get("/v1/models")
        assert r.status_code == 200
        assert r.json()["data"][0]["id"] == "fake-model"


def test_non_stream_response_has_no_metrics():
    with _client() as client:
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8, "temperature": 0,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert "metrics" not in body
        assert body["choices"][0]["message"]["content"]
        assert body["usage"]["prompt_tokens"] == 3


def test_api_key_required_for_v1():
    with _client(api_key="secret") as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/models").status_code == 401
        headers = {"Authorization": "Bearer secret"}
        assert client.get("/v1/models", headers=headers).status_code == 200
        r = client.post("/v1/chat/completions", headers=headers, json={
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8, "temperature": 0,
        })
        assert r.status_code == 200, r.text


def test_invalid_sampling_maps_to_400():
    with _client() as client:
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 0,
        })
        assert r.status_code == 400