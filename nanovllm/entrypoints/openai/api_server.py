"""Minimal OpenAI-compatible chat completion server."""
import argparse
import json
import time
import uuid

from nanovllm.async_engine import AsyncLLMEngine
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
        lines = []
        for name, value in engine.metrics().items():
            lines.append(f"nanovllm_{name} {value}")
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
        handle = await engine.add_request(
            prompt_ids,
            SamplingParams(temperature=temperature, max_tokens=max_tokens),
        )
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        async def events():
            token_ids = []
            previous = ""
            finish_reason = "stop"
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
                        text = engine.tokenizer.decode(token_ids)
                        matched = next((marker for marker in stop if marker in text), None)
                        if matched is not None:
                            text = text.split(matched, 1)[0]
                            engine.cancel(handle.seq_id)
                        delta = text[len(previous):]
                        previous = text
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
                        final = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                        }
                        yield f"data: {json.dumps(final)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
            finally:
                engine.cancel(handle.seq_id)

        if body.get("stream", False):
            return StreamingResponse(events(), media_type="text/event-stream")

        token_ids = []
        metrics = None
        try:
            while True:
                event = await engine.next_event(handle)
                if event.error:
                    raise HTTPException(500, event.error)
                if event.token_id is not None:
                    token_ids.append(event.token_id)
                text = engine.tokenizer.decode(token_ids)
                matched = next((marker for marker in stop if marker in text), None)
                if matched is not None:
                    text = text.split(matched, 1)[0]
                    engine.cancel(handle.seq_id)
                    break
                if event.finished:
                    metrics = event.metrics
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
                "finish_reason": "stop",
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
    parser.add_argument("--moe-backend", choices=("loop", "grouped_mm"), default="grouped_mm")
    parser.add_argument("--expert-placement", choices=("linear", "round_robin", "profiled"), default="linear")
    parser.add_argument("--expert-map-path")
    parser.add_argument("--scheduler-policy", choices=("fcfs", "stall_free", "stall_free_lpm"), default="stall_free")
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--metrics-path")
    args = parser.parse_args()

    import uvicorn

    engine = AsyncLLMEngine(
        args.model,
        tensor_parallel_size=2,
        expert_parallel_size=2,
        moe_backend=args.moe_backend,
        expert_placement=args.expert_placement,
        expert_map_path=args.expert_map_path,
        scheduler_policy=args.scheduler_policy,
        prefill_chunk_size=args.prefill_chunk_size,
        metrics_path=args.metrics_path,
        enforce_eager=True,
    )
    model_name = args.served_model_name or args.model.rstrip("/").rsplit("/", 1)[-1]
    uvicorn.run(create_app(engine, model_name), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
