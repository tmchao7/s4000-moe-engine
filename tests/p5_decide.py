"""CPU-only boundary tests for the frozen P5/P6 decision gates.

Run from the repository root so that ``scripts`` is importable:

    python3 tests/p5_decide.py
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

# Load by file path: some environments ship an unrelated site-packages
# ``scripts`` package that shadows this repository's namespace package.
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p5_decide.py"
_spec = importlib.util.spec_from_file_location("p5_decide", _SCRIPT)
p5_decide = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p5_decide)

_decide_attention = p5_decide.decide_attention
_decide_moe = p5_decide.decide_moe
_decide_graph = p5_decide.decide_graph
_decision_payload = p5_decide._decision_payload
_sha256_file = p5_decide.sha256_file

decide_attention = _decide_attention
decide_moe = _decide_moe
decide_graph = _decide_graph
sha256_file = _sha256_file


def attention_payload(tpot_ratio=0.85, throughput_ratio=1.05, cv=0.02, mismatch=0):
    summary = []
    for backend, tpot, throughput in (
        ("sdpa", 0.5, 100.0),
        ("triton_paged", 0.5 * tpot_ratio, 100.0 * throughput_ratio),
    ):
        for batch in (16, 32):
            summary.append({
                "attention_backend": backend,
                "batch": batch,
                "context": 2048,
                "throughput_median": throughput,
                "throughput_cv": cv,
                "ttft_p95_median_s": 1.0,
                "tpot_p95_median_s": tpot,
                "itl_p95_median_s": tpot,
                "token_mismatch_requests": mismatch if backend == "triton_paged" else 0,
                "token_identity": mismatch == 0,
            })
    return {"summary": summary}


def moe_payload(throughput_ratio=1.10, tpot_ratio=0.98, cv=0.02, token_mismatch=False):
    records = []
    summary = []
    for backend, multiplier in (("grouped_mm", 1.0), ("vllm_fused", throughput_ratio)):
        for batch in (16, 32):
            tokens = [1, 2, 3]
            if token_mismatch and backend == "vllm_fused":
                tokens = [1, 2, 9]
            records.append({
                "backend": backend,
                "batch": batch,
                "repeat": 0,
                "cache_mode": "none",
                "requests": [{"token_ids": tokens}],
            })
            summary.append({
                "backend": backend,
                "batch": batch,
                "cache_mode": "none",
                "throughput_median": 100.0 * multiplier,
                "throughput_cv": cv,
                "ttft_p95_median_s": 1.0,
                "tpot_p95_median_s": 0.4 * tpot_ratio if backend == "vllm_fused" else 0.4,
                "itl_p95_median_s": 0.4,
            })
    return {"records": records, "summary": summary, "repeats": 1}


def graph_payload(b1_ratio=1.20, b8_ratio=1.06, b32_ratio=1.01, cv=0.02,
                  token_identity=True, memory_delta_gib=1.0):
    summary = []
    for graph in (False, True):
        for batch, ratio in ((1, b1_ratio), (8, b8_ratio), (32, b32_ratio), (64, b32_ratio)):
            summary.append({
                "graph": graph,
                "batch": batch,
                "throughput_median": 10.0 * (ratio if graph else 1.0),
                "throughput_cv": cv,
                "token_identity": token_identity,
            })
    return {"summary": summary, "memory_delta_gib": memory_delta_gib}


def test_attention_gates():
    assert decide_attention(attention_payload())["decision"] == "adopt_triton_paged"
    assert decide_attention(attention_payload(tpot_ratio=0.94))["decision"] == "keep_sdpa"
    assert decide_attention(attention_payload(throughput_ratio=0.99))["decision"] == "keep_sdpa"
    assert decide_attention(attention_payload(cv=0.06))["decision"] == "keep_sdpa"
    mismatched = decide_attention(attention_payload(mismatch=2))
    assert mismatched["decision"] == "keep_sdpa"
    assert mismatched["token_identity"] is False


def test_moe_gates():
    assert decide_moe(moe_payload())["decision"] == "adopt_vllm_fused"
    assert decide_moe(moe_payload(throughput_ratio=1.07))["decision"] == "keep_grouped_mm"
    assert decide_moe(moe_payload(tpot_ratio=1.02))["decision"] == "keep_grouped_mm"
    assert decide_moe(moe_payload(cv=0.06))["decision"] == "keep_grouped_mm"
    mismatched = decide_moe(moe_payload(token_mismatch=True))
    assert mismatched["decision"] == "keep_grouped_mm"
    assert mismatched["token_identity"] is False


def test_graph_gates():
    assert decide_graph(graph_payload())["decision"] == "adopt_cudagraph"
    assert decide_graph(graph_payload(b1_ratio=1.14))["decision"] == "keep_eager"
    assert decide_graph(graph_payload(b8_ratio=1.04))["decision"] == "keep_eager"
    assert decide_graph(graph_payload(b32_ratio=0.97))["decision"] == "keep_eager"
    assert decide_graph(graph_payload(memory_delta_gib=3.0))["decision"] == "keep_eager"
    assert decide_graph(graph_payload(token_identity=False))["decision"] == "keep_eager"


def test_evidence_hash():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "ab.json"
        path.write_text(json.dumps({"summary": []}), encoding="utf-8")
        decision = _decision_payload("attention", path, {"x": 1}, {"decision": "keep_sdpa"})
        assert decision["evidence_sha256"] == sha256_file(path)
        assert len(decision["evidence_sha256"]) == 64


def main():
    test_attention_gates()
    test_moe_gates()
    test_graph_gates()
    test_evidence_hash()
    print("[p5-decide-test] attention/moe/graph gate boundaries passed")


if __name__ == "__main__":
    main()
