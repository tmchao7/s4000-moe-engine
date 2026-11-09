"""Minimal OpenAI-compatible chat/completion server."""
import argparse
import faulthandler
import json
import os
import time
import uuid

from nanovllm import __version__
from nanovllm.async_engine import AsyncLLMEngine
from nanovllm.incremental_detokenizer import IncrementalDetokenizer
from nanovllm.sampling_params import SamplingParams


def _parse_sampling_kwargs(body: dict, HTTPException, logprobs_int: bool) -> dict:
    """Shared OpenAI -> SamplingParams translation (P22-A).

    logprobs_int selects the two OpenAI shapes: /v1/completions carries an
    integer `logprobs`, /v1/chat/completions a boolean `logprobs` plus an
    integer `top_logprobs`.  Invalid values surface as SamplingParams
    ValueError at construction — the endpoints map that to 400.
    """
    def num(key, cast, default):
        try:
            return cast(body.get(key, default))
        except (TypeError, ValueError):
            raise HTTPException(400, f"{key} must be a number")

    kwargs = {
        "top_p": num("top_p", float, 1.0),
        "top_k": num("top_k", int, -1),
        "min_p": num("min_p", float, 0.0),
        "presence_penalty": num("presence_penalty", float, 0.0),
        "frequency_penalty": num("frequency_penalty", float, 0.0),
        "repetition_penalty": num("repetition_penalty", float, 1.0),
        "skip_special_tokens": bool(body.get("skip_special_tokens", True)),
    }
    stop_ids = body.get("stop_token_ids")
    if stop_ids is not None:
        if not isinstance(stop_ids, list) or not all(
                isinstance(t, int) and not isinstance(t, bool) for t in stop_ids):
            raise HTTPException(400, "stop_token_ids must be a list of integers")
        kwargs["stop_token_ids"] = stop_ids
    if logprobs_int:
        if body.get("logprobs") is not None:
            kwargs["logprobs"] = num("logprobs", int, 0)
    elif body.get("logprobs"):
        kwargs["logprobs"] = max(1, num("top_logprobs", int, 0))
    # P24 structured output: response_format json_schema (OpenAI shape, with
    # the vLLM-style direct-schema tolerance) or the vLLM guided_* aliases.
    rf = body.get("response_format")
    if rf is not None:
        if not isinstance(rf, dict) or rf.get("type") != "json_schema":
            raise HTTPException(400, "response_format.type must be 'json_schema'")
        js = rf.get("json_schema")
        schema = js.get("schema") if isinstance(js, dict) else None
        if schema is None and isinstance(js, dict) and js:
            schema = js                     # bare schema object tolerated
        if not isinstance(schema, (dict, str)):
            raise HTTPException(400, "response_format.json_schema.schema must be an object")
        kwargs["guided_json"] = schema
    if body.get("guided_json") is not None:
        kwargs["guided_json"] = body["guided_json"]
    if body.get("guided_regex") is not None:
        kwargs["guided_regex"] = body["guided_regex"]
    return kwargs


def _token_str(tokenizer, cache: dict, token_id: int) -> str:
    text = cache.get(token_id)
    if text is None:
        text = tokenizer.decode([token_id])
        cache[token_id] = text
    return text


def _top_lp(tokenizer, cache: dict, token_id: int, logprob: float) -> dict:
    text = _token_str(tokenizer, cache, token_id)
    return {"token": text, "logprob": logprob, "bytes": list(text.encode("utf-8"))}


def create_app(engine: AsyncLLMEngine, model_name: str, api_key: str | None = None,
               version: str | None = None):
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
    except ImportError as exc:
        raise RuntimeError("install the serve dependencies: pip install -r requirements-serve.txt") from exc

    app = FastAPI(title="s4000-moe-engine")

    if api_key:
        @app.middleware("http")
        async def require_api_key(request: Request, call_next):
            if request.url.path.startswith("/v1/") and \
                    request.headers.get("authorization") != f"Bearer {api_key}":
                return JSONResponse(
                    status_code=401,
                    content={"error": {"message": "invalid API key",
                                       "type": "invalid_request_error"}},
                )
            return await call_next(request)

    @app.on_event("shutdown")
    async def shutdown():
        await engine.close()

    @app.get("/health")
    async def health():
        payload = {"status": "ok", "model": model_name}
        if version:
            payload["version"] = version
        return payload

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "local"}]}

    @app.get("/metrics")
    async def metrics():
        if hasattr(engine, "prometheus_text"):
            return PlainTextResponse(engine.prometheus_text())
        lines = []
        for name, value in engine.metrics().items():
            metric = f"nanovllm_{name}"
            metric_type = "counter" if name.endswith("_total") else "gauge"
            lines.extend((
                f"# HELP {metric} nano-vLLM engine metric {name}.",
                f"# TYPE {metric} {metric_type}",
                f"{metric} {value}",
            ))
        return PlainTextResponse("\n".join(lines) + "\n")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        if body.get("n", 1) != 1:
            raise HTTPException(400, "only n=1 is supported")
        unsupported = [key for key in ("tools", "tool_choice") if body.get(key) not in (None, False)]
        if unsupported:
            raise HTTPException(400, f"unsupported parameters: {', '.join(unsupported)}")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(400, "messages must be a non-empty list")
        try:
            max_tokens = int(body.get("max_tokens", 64))
            temperature = float(body.get("temperature", 1.0))
        except (TypeError, ValueError):
            raise HTTPException(400, "max_tokens/temperature must be numbers")
        # P22-A: top_p/top_k/min_p/penalties/stop_token_ids/logprobs ride
        # through; SamplingParams validation failures map to 400 below.
        sampling_kwargs = _parse_sampling_kwargs(body, HTTPException, logprobs_int=False)
        # P16-C extension: scheduling tier (0 = highest).  Non-standard field,
        # ignored by stock OpenAI clients; equal priorities keep FIFO.  NOTE:
        # ordering applies under stall_free policies; with the fcfs default
        # only preemption-victim selection honors it.
        try:
            priority = int(body.get("priority", 0))
        except (TypeError, ValueError):
            raise HTTPException(400, "priority must be an integer")
        if max_tokens <= 0 or temperature < 0 or priority < 0:
            raise HTTPException(400, "max_tokens must be positive; temperature/priority non-negative")
        stop = body.get("stop", [])
        if isinstance(stop, str):
            stop = [stop]
        if not isinstance(stop, list) or not all(isinstance(item, str) for item in stop):
            raise HTTPException(400, "stop must be a string or list of strings")
        stream_options = body.get("stream_options") or {}
        if not isinstance(stream_options, dict):
            raise HTTPException(400, "stream_options must be an object")
        include_usage = bool(stream_options.get("include_usage", False))
        include_metrics = bool(stream_options.get("include_metrics", False))

        try:
            prompt_ids = engine.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        except Exception as exc:
            raise HTTPException(400, f"invalid chat messages: {exc}") from exc
        # transformers 5.x returns a BatchEncoding (and may batch the ids);
        # normalise to a flat list of ints before it reaches Sequence.
        if hasattr(prompt_ids, "keys"):
            prompt_ids = prompt_ids["input_ids"]
        if hasattr(prompt_ids, "tolist"):
            prompt_ids = prompt_ids.tolist()
        if prompt_ids and isinstance(prompt_ids[0], (list, tuple)):
            prompt_ids = prompt_ids[0]
        prompt_ids = [int(token_id) for token_id in prompt_ids]
        if not prompt_ids:
            raise HTTPException(400, "chat template produced an empty prompt")
        try:
            sp = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                priority=priority, **sampling_kwargs)
            handle = await engine.add_request(prompt_ids, sp)
        except OverflowError as exc:
            raise HTTPException(429, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        tokenizer = engine.tokenizer
        lp_cache = {}

        def lp_entry(token_id, logprobs):
            text = _token_str(tokenizer, lp_cache, token_id)
            return {
                "token": text,
                "logprob": logprobs["chosen"],
                "bytes": list(text.encode("utf-8")),
                "top_logprobs": [
                    _top_lp(tokenizer, lp_cache, tid, val) for tid, val in logprobs["top"]
                ],
            }

        async def events():
            token_ids = []
            decoder = IncrementalDetokenizer(tokenizer, skip_special_tokens=sp.skip_special_tokens)
            finish_reason = "length"
            output_text = ""
            try:
                while True:
                    if await request.is_disconnected():
                        engine.cancel(handle.seq_id)
                        return
                    event = await engine.next_event(handle)
                    if event.error:
                        payload = {"error": {"message": event.error, "type": "engine_error"}}
                        yield f"data: {json.dumps(payload)}\n\n"
                        return
                    if event.cancelled:
                        return
                    if event.token_id is not None:
                        token_ids.append(event.token_id)
                        delta = decoder.push(event.token_id)
                        output_text += delta
                        matched = next((marker for marker in stop if marker in output_text), None)
                        if matched is not None:
                            visible = output_text.split(matched, 1)[0]
                            retract = len(output_text) - len(visible)
                            delta = delta[:-retract] if retract and retract <= len(delta) else ""
                            output_text = visible
                            engine.cancel(handle.seq_id)
                            finish_reason = "stop"
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                        }
                        if event.logprobs is not None:
                            chunk["choices"][0]["logprobs"] = {"content": [lp_entry(event.token_id, event.logprobs)]}
                        if delta:
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        if matched is not None:
                            event.finished = True
                    if event.finished:
                        if event.metrics and event.metrics.get("finish_reason"):
                            finish_reason = event.metrics["finish_reason"]
                        final = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                        }
                        if include_metrics and event.metrics:
                            final["metrics"] = {
                                key: event.metrics[key]
                                for key in (
                                    "prompt_tokens", "output_tokens", "queue_s",
                                    "ttft_s", "e2e_s", "tpot_s", "mean_tpot_s",
                                    "itls_s", "finish_reason",
                                )
                                if key in event.metrics
                            }
                        yield f"data: {json.dumps(final)}\n\n"
                        if include_usage:
                            usage = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_name,
                                "choices": [],
                                "usage": {
                                    "prompt_tokens": len(prompt_ids),
                                    "completion_tokens": len(token_ids),
                                    "total_tokens": len(prompt_ids) + len(token_ids),
                                },
                            }
                            yield f"data: {json.dumps(usage)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
            finally:
                engine.cancel(handle.seq_id)

        if body.get("stream", False):
            return StreamingResponse(events(), media_type="text/event-stream")

        token_ids = []
        decoder = IncrementalDetokenizer(tokenizer, skip_special_tokens=sp.skip_special_tokens)
        text = ""
        metrics = None
        finish_reason = "length"
        lp_entries = []
        try:
            while True:
                event = await engine.next_event(handle)
                if event.error:
                    raise HTTPException(500, event.error)
                if event.token_id is not None:
                    token_ids.append(event.token_id)
                    text += decoder.push(event.token_id)
                    if event.logprobs is not None:
                        lp_entries.append(lp_entry(event.token_id, event.logprobs))
                matched = next((marker for marker in stop if marker in text), None)
                if matched is not None:
                    text = text.split(matched, 1)[0]
                    engine.cancel(handle.seq_id)
                    finish_reason = "stop"
                    break
                if event.finished:
                    metrics = event.metrics
                    if metrics and metrics.get("finish_reason"):
                        finish_reason = metrics["finish_reason"]
                    break
        finally:
            engine.cancel(handle.seq_id)
        return JSONResponse({
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
                "logprobs": {"content": lp_entries} if lp_entries else None,
            }],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(token_ids),
                "total_tokens": len(prompt_ids) + len(token_ids),
            },
        })

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        if body.get("n", 1) != 1:
            raise HTTPException(400, "only n=1 is supported")
        unsupported = [key for key in ("tools", "tool_choice", "echo", "suffix", "best_of")
                       if body.get(key) not in (None, False)]
        if unsupported:
            raise HTTPException(400, f"unsupported parameters: {', '.join(unsupported)}")
        tokenizer = engine.tokenizer
        lp_cache = {}
        prompt = body.get("prompt")
        if isinstance(prompt, str):
            prompt_ids = tokenizer.encode(prompt)
        elif isinstance(prompt, list) and prompt and all(
                isinstance(t, int) and not isinstance(t, bool) for t in prompt):
            prompt_ids = [int(t) for t in prompt]
        else:
            raise HTTPException(400, "prompt must be a string or a list of token ids")
        if not prompt_ids:
            raise HTTPException(400, "prompt produced an empty token list")
        max_tokens = body.get("max_tokens", 64)
        temperature = body.get("temperature", 1.0)
        try:
            max_tokens = int(max_tokens)
            temperature = float(temperature)
        except (TypeError, ValueError):
            raise HTTPException(400, "max_tokens/temperature must be numbers")
        try:
            priority = int(body.get("priority", 0))
        except (TypeError, ValueError):
            raise HTTPException(400, "priority must be an integer")
        if max_tokens <= 0 or temperature < 0 or priority < 0:
            raise HTTPException(400, "max_tokens must be positive; temperature/priority non-negative")
        sampling_kwargs = _parse_sampling_kwargs(body, HTTPException, logprobs_int=True)
        stop = body.get("stop", [])
        if isinstance(stop, str):
            stop = [stop]
        if not isinstance(stop, list) or not all(isinstance(item, str) for item in stop):
            raise HTTPException(400, "stop must be a string or list of strings")
        stream_options = body.get("stream_options") or {}
        if not isinstance(stream_options, dict):
            raise HTTPException(400, "stream_options must be an object")
        include_usage = bool(stream_options.get("include_usage", False))
        try:
            sp = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                priority=priority, **sampling_kwargs)
            handle = await engine.add_request(prompt_ids, sp)
        except OverflowError as exc:
            raise HTTPException(429, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        completion_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        def lp_arrays(entries, offsets):
            """Legacy completions logprobs shape for a batch of token events."""
            return {
                "tokens": [_token_str(tokenizer, lp_cache, e[0]) for e in entries],
                "token_logprobs": [e[1]["chosen"] for e in entries],
                "top_logprobs": [
                    {_token_str(tokenizer, lp_cache, tid): val for tid, val in e[1]["top"]}
                    for e in entries
                ],
                "text_offset": offsets,
            }

        async def events():
            token_ids = []
            decoder = IncrementalDetokenizer(tokenizer, skip_special_tokens=sp.skip_special_tokens)
            finish_reason = "length"
            output_text = ""
            try:
                while True:
                    if await request.is_disconnected():
                        engine.cancel(handle.seq_id)
                        return
                    event = await engine.next_event(handle)
                    if event.error:
                        payload = {"error": {"message": event.error, "type": "engine_error"}}
                        yield f"data: {json.dumps(payload)}\n\n"
                        return
                    if event.cancelled:
                        return
                    if event.token_id is not None:
                        lp_token = None
                        offset = len(output_text)
                        if event.logprobs is not None:
                            lp_token = (event.token_id, event.logprobs)
                        token_ids.append(event.token_id)
                        delta = decoder.push(event.token_id)
                        output_text += delta
                        matched = next((marker for marker in stop if marker in output_text), None)
                        if matched is not None:
                            visible = output_text.split(matched, 1)[0]
                            retract = len(output_text) - len(visible)
                            delta = delta[:-retract] if retract and retract <= len(delta) else ""
                            output_text = visible
                            engine.cancel(handle.seq_id)
                            finish_reason = "stop"
                        chunk = {
                            "id": completion_id,
                            "object": "text_completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{"index": 0, "text": delta, "finish_reason": None}],
                        }
                        if lp_token is not None:
                            chunk["choices"][0]["logprobs"] = lp_arrays([lp_token], [offset])
                        if delta:
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        if matched is not None:
                            event.finished = True
                    if event.finished:
                        if event.metrics and event.metrics.get("finish_reason"):
                            finish_reason = event.metrics["finish_reason"]
                        final = {
                            "id": completion_id,
                            "object": "text_completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{"index": 0, "text": "", "finish_reason": finish_reason}],
                        }
                        if include_usage:
                            final["usage"] = {
                                "prompt_tokens": len(prompt_ids),
                                "completion_tokens": len(token_ids),
                                "total_tokens": len(prompt_ids) + len(token_ids),
                            }
                        yield f"data: {json.dumps(final)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
            finally:
                engine.cancel(handle.seq_id)

        if body.get("stream", False):
            return StreamingResponse(events(), media_type="text/event-stream")

        token_ids = []
        decoder = IncrementalDetokenizer(tokenizer, skip_special_tokens=sp.skip_special_tokens)
        text = ""
        metrics = None
        finish_reason = "length"
        lp_entries = []
        lp_offsets = []
        try:
            while True:
                event = await engine.next_event(handle)
                if event.error:
                    raise HTTPException(500, event.error)
                if event.token_id is not None:
                    lp_offsets.append(len(text))
                    token_ids.append(event.token_id)
                    text += decoder.push(event.token_id)
                    if event.logprobs is not None:
                        lp_entries.append((event.token_id, event.logprobs))
                matched = next((marker for marker in stop if marker in text), None)
                if matched is not None:
                    text = text.split(matched, 1)[0]
                    engine.cancel(handle.seq_id)
                    finish_reason = "stop"
                    break
                if event.finished:
                    metrics = event.metrics
                    if metrics and metrics.get("finish_reason"):
                        finish_reason = metrics["finish_reason"]
                    break
        finally:
            engine.cancel(handle.seq_id)
        return JSONResponse({
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "text": text,
                "finish_reason": finish_reason,
                "logprobs": lp_arrays(lp_entries, lp_offsets) if lp_entries else None,
            }],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(token_ids),
                "total_tokens": len(prompt_ids) + len(token_ids),
            },
        })

    return app


def _env(name, default=None):
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got {value!r}") from exc


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a number, got {value!r}") from exc


def _is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="s4000-server",
        description="s4000-moe-engine OpenAI-compatible server")
    parser.add_argument(
        "--model",
        default=_env("NANOVLLM_MODEL"),
        help="model directory (or set NANOVLLM_MODEL)",
    )
    parser.add_argument("--served-model-name",
                        default=_env("NANOVLLM_SERVED_MODEL_NAME"))
    parser.add_argument("--host", default=_env("NANOVLLM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=_env_int("NANOVLLM_PORT", 8000))
    parser.add_argument(
        "--moe-backend",
        choices=("loop", "grouped_mm", "fused_grouped_mm", "vllm_fused"),
        default=_env("NANOVLLM_MOE_BACKEND", "grouped_mm"),
    )
    parser.add_argument(
        "--decode-moe-backend",
        choices=("loop", "grouped_mm", "fused_grouped_mm", "bmm_dense", "vllm_fused"),
        default=_env("NANOVLLM_DECODE_MOE_BACKEND", "bmm_dense"),
        help="decode-only MoE backend; product default bmm_dense",
    )
    parser.add_argument("--attention-backend",
                        choices=("sdpa", "triton_paged"),
                        default=_env("NANOVLLM_ATTENTION_BACKEND", "triton_paged"))
    parser.add_argument("--expert-placement",
                        choices=("linear", "round_robin", "profiled"),
                        default=_env("NANOVLLM_EXPERT_PLACEMENT", "linear"))
    parser.add_argument("--expert-map-path",
                        default=_env("NANOVLLM_EXPERT_MAP_PATH"))
    parser.add_argument("--scheduler-policy",
                        choices=("fcfs", "stall_free", "stall_free_lpm"),
                        default=_env("NANOVLLM_SCHEDULER_POLICY", "stall_free"))
    parser.add_argument("--prefill-chunk-size", type=int,
                        default=_env_int("NANOVLLM_PREFILL_CHUNK_SIZE", 256))
    parser.add_argument("--metrics-path", default=_env("NANOVLLM_METRICS_PATH"))
    parser.add_argument("--max-queue-size", type=int,
                        default=_env_int("NANOVLLM_MAX_QUEUE_SIZE", 1024))
    parser.add_argument("--max-model-len", type=int,
                        default=_env_int("NANOVLLM_MAX_MODEL_LEN", 4096),
                        help="context limit; also bounds warmup and the KV table width")
    parser.add_argument("--max-num-seqs", type=int,
                        default=_env_int("NANOVLLM_MAX_NUM_SEQS", None),
                        help="override Config.max_num_seqs (default 512)")
    parser.add_argument("--tensor-parallel-size", type=int,
                        default=_env_int("NANOVLLM_TP_SIZE", 2))
    parser.add_argument("--expert-parallel-size", type=int,
                        default=_env_int("NANOVLLM_EP_SIZE", 0),
                        help="0 = same as tensor-parallel-size")
    parser.add_argument("--gpu-memory-utilization", type=float,
                        default=_env_float("NANOVLLM_GPU_MEM_UTIL", 0.9))
    parser.add_argument("--api-key", default=_env("NANOVLLM_API_KEY"),
                        help="optional Bearer token; required for non-loopback host")
    parser.add_argument("--allow-insecure", action="store_true",
                        default=_env("NANOVLLM_ALLOW_INSECURE", "0") == "1",
                        help="allow non-loopback bind without an API key")
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction,
                        default=_env("NANOVLLM_ENFORCE_EAGER", "1") != "0")
    parser.add_argument("--scheduler-overlap", action=argparse.BooleanOptionalAction,
                        default=_env("NANOVLLM_SCHEDULER_OVERLAP", "0") == "1",
                        help="advanced one-batch-in-flight pipeline (opt-in)")
    parser.add_argument("--kv-offload", action=argparse.BooleanOptionalAction,
                        default=_env("NANOVLLM_KV_OFFLOAD", "0") == "1",
                        help="swap LRU-evicted KV blocks to pinned host memory "
                             "and copy back on prefix revival (opt-in; bf16 "
                             "KV cache only)")
    parser.add_argument("--kv-offload-host-blocks", type=int,
                        default=_env_int("NANOVLLM_KV_OFFLOAD_HOST_BLOCKS", 256),
                        help="host pool capacity in KV blocks (0 disables the "
                             "pool; offload then degrades to plain eviction)")
    parser.add_argument("--log-level", default=_env("NANOVLLM_LOG_LEVEL", "info"),
                        choices=("critical", "error", "warning", "info", "debug", "trace"))
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def validate_args(parser, args):
    if not args.model:
        parser.error("--model or NANOVLLM_MODEL is required")
    if not os.path.isdir(args.model):
        parser.error(f"model directory not found: {args.model}")
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be >= 1")
    if args.expert_parallel_size not in (0, args.tensor_parallel_size):
        parser.error("--expert-parallel-size must be 0 or equal to tensor-parallel-size")
    if args.max_queue_size <= 0:
        parser.error("--max-queue-size must be positive")
    if args.api_key is not None:
        args.api_key = args.api_key.strip() or None
    if not _is_loopback(args.host) and not args.api_key and not args.allow_insecure:
        parser.error("refusing non-loopback bind without --api-key; set "
                     "NANOVLLM_API_KEY or pass --allow-insecure for a trusted network")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    if os.environ.get("NANOVLLM_TRACE_INIT") == "1":
        # Startup is silent between weight load and KV allocation; a hung rank
        # otherwise leaves no stack to look at.
        faulthandler.dump_traceback_later(180, repeat=True)

    import uvicorn

    if args.max_num_seqs is not None:
        print(f"[init] max_num_seqs={args.max_num_seqs}", flush=True)
    print(
        f"[serve] model={args.model} tp={args.tensor_parallel_size} "
        f"ep={args.expert_parallel_size} host={args.host} port={args.port} "
        f"decode={args.decode_moe_backend}",
        flush=True,
    )
    engine = AsyncLLMEngine(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        expert_parallel_size=args.expert_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        moe_backend=args.moe_backend,
        decode_moe_backend=args.decode_moe_backend,
        attention_backend=args.attention_backend,
        expert_placement=args.expert_placement,
        expert_map_path=args.expert_map_path,
        scheduler_policy=args.scheduler_policy,
        prefill_chunk_size=args.prefill_chunk_size,
        metrics_path=args.metrics_path,
        max_queue_size=args.max_queue_size,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        scheduler_overlap=args.scheduler_overlap,
        kv_offload_enabled=args.kv_offload,
        kv_offload_host_blocks=args.kv_offload_host_blocks,
        **({"max_num_seqs": args.max_num_seqs} if args.max_num_seqs is not None else {}),
    )
    model_name = args.served_model_name or args.model.rstrip("/").rsplit("/", 1)[-1]
    uvicorn.run(
        create_app(engine, model_name, api_key=args.api_key, version=__version__),
        host=args.host, port=args.port, log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
