"""Minimal OpenAI-compatible chat completion server."""
import argparse
import faulthandler
import json
import os
import time
import uuid

from nanovllm.async_engine import AsyncLLMEngine
from nanovllm.incremental_detokenizer import IncrementalDetokenizer
from nanovllm.sampling_params import SamplingParams


def create_app(engine: AsyncLLMEngine, model_name: str):
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
    except ImportError as exc:
        raise RuntimeError("install nano-vllm[serve] to run the API server") from exc

    app = FastAPI(title="moe-server-s4000")

    @app.on_event("shutdown")
    async def shutdown():
        await engine.close()

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": model_name}

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
        unsupported = [key for key in ("tools", "tool_choice", "logprobs") if body.get(key) not in (None, False)]
        if unsupported:
            raise HTTPException(400, f"unsupported parameters: {', '.join(unsupported)}")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(400, "messages must be a non-empty list")
        max_tokens = int(body.get("max_tokens", 64))
        temperature = float(body.get("temperature", 1.0))
        if max_tokens <= 0 or temperature < 0:
            raise HTTPException(400, "max_tokens must be positive and temperature non-negative")
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
            handle = await engine.add_request(
                prompt_ids,
                SamplingParams(temperature=temperature, max_tokens=max_tokens),
            )
        except OverflowError as exc:
            raise HTTPException(429, str(exc)) from exc
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        async def events():
            token_ids = []
            decoder = IncrementalDetokenizer(engine.tokenizer)
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
        decoder = IncrementalDetokenizer(engine.tokenizer)
        text = ""
        metrics = None
        finish_reason = "length"
        try:
            while True:
                event = await engine.next_event(handle)
                if event.error:
                    raise HTTPException(500, event.error)
                if event.token_id is not None:
                    token_ids.append(event.token_id)
                    text += decoder.push(event.token_id)
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
            }],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(token_ids),
                "total_tokens": len(prompt_ids) + len(token_ids),
            },
            "metrics": metrics,
        })

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--moe-backend", choices=("loop", "grouped_mm", "fused_grouped_mm", "vllm_fused"), default="grouped_mm")
    parser.add_argument(
        "--decode-moe-backend",
        choices=("loop", "grouped_mm", "fused_grouped_mm", "bmm_dense", "vllm_fused"),
        default=None,
        help="decode-only MoE backend (e.g. bmm_dense); default = same as --moe-backend",
    )
    parser.add_argument("--attention-backend", choices=("sdpa", "triton_paged"), default="triton_paged")
    parser.add_argument("--expert-placement", choices=("linear", "round_robin", "profiled"), default="linear")
    parser.add_argument("--expert-map-path")
    parser.add_argument("--scheduler-policy", choices=("fcfs", "stall_free", "stall_free_lpm"), default="stall_free")
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--metrics-path")
    parser.add_argument("--max-queue-size", type=int, default=1024)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="context limit; also bounds warmup and the KV block table width",
    )
    args = parser.parse_args()

    if os.environ.get("NANOVLLM_TRACE_INIT") == "1":
        # Startup is silent between weight load and KV allocation; a hung rank
        # otherwise leaves no stack to look at.
        faulthandler.dump_traceback_later(180, repeat=True)

    import uvicorn

    engine = AsyncLLMEngine(
        args.model,
        tensor_parallel_size=2,
        expert_parallel_size=2,
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
        enforce_eager=True,
    )
    flip_backend = os.environ.get("NANOVLLM_ATTENTION_FLIP_AFTER_INIT")
    if flip_backend:
        # Diagnostic path: start on the known-good backend and switch once the
        # engine is fully initialised.  Same broadcast the A/B harness uses, so
        # the numerics are the ones verified in L1-L4.
        assert flip_backend in ("sdpa", "triton_paged"), flip_backend
        print(f"[init] flipping attention backend to {flip_backend} after init", flush=True)
        engine.engine.model_runner.call("set_attention_backend", flip_backend)
        print("[init] attention backend flip done", flush=True)
    model_name = args.served_model_name or args.model.rstrip("/").rsplit("/", 1)[-1]
    uvicorn.run(create_app(engine, model_name), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
