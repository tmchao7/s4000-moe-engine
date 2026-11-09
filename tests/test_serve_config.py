"""Product CLI/env contract tests (CPU; no GPU)."""
import os
import sys

import pytest

pytest.importorskip("tqdm")
pytest.importorskip("torch")

from nanovllm.entrypoints.openai import api_server


def _parse(monkeypatch, argv, env=None):
    for key in list(os.environ):
        if key.startswith("NANOVLLM_"):
            monkeypatch.delenv(key, raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    return api_server.build_parser().parse_args(argv)


def test_defaults_are_product_defaults(monkeypatch, tmp_path):
    args = _parse(monkeypatch, ["--model", str(tmp_path)])
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.tensor_parallel_size == 2
    assert args.expert_parallel_size == 0
    assert args.decode_moe_backend == "bmm_dense"
    assert args.scheduler_policy == "stall_free"
    assert args.enforce_eager is True


def test_env_is_used_and_cli_wins(monkeypatch, tmp_path):
    env = {
        "NANOVLLM_MODEL": str(tmp_path),
        "NANOVLLM_PORT": "9001",
        "NANOVLLM_TP_SIZE": "1",
        "NANOVLLM_EP_SIZE": "1",
        "NANOVLLM_DECODE_MOE_BACKEND": "loop",
    }
    args = _parse(monkeypatch, [], env)
    assert args.model == str(tmp_path)
    assert args.port == 9001
    assert args.tensor_parallel_size == 1
    assert args.expert_parallel_size == 1
    assert args.decode_moe_backend == "loop"

    args = _parse(monkeypatch, ["--port", "9002"], env)
    assert args.port == 9002


def test_missing_model_is_rejected(monkeypatch):
    parser = api_server.build_parser()
    args = _parse(monkeypatch, [])
    with pytest.raises(SystemExit):
        api_server.validate_args(parser, args)


def test_non_loopback_requires_key(monkeypatch, tmp_path):
    parser = api_server.build_parser()
    args = _parse(monkeypatch, ["--model", str(tmp_path), "--host", "0.0.0.0"])
    with pytest.raises(SystemExit):
        api_server.validate_args(parser, args)

    args = _parse(monkeypatch, ["--model", str(tmp_path), "--host", "0.0.0.0",
                               "--api-key", "secret"])
    api_server.validate_args(parser, args)
    assert args.api_key == "secret"


def test_ep_must_match_tp(monkeypatch, tmp_path):
    parser = api_server.build_parser()
    args = _parse(monkeypatch, ["--model", str(tmp_path), "--tensor-parallel-size", "2",
                               "--expert-parallel-size", "1"])
    with pytest.raises(SystemExit):
        api_server.validate_args(parser, args)


def test_help_does_not_advertise_benchmarks(monkeypatch, capsys):
    parser = api_server.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    out = capsys.readouterr().out
    assert "benchmarks/" not in out