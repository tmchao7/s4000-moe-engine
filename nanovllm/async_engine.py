"""Single-model background engine used by the OpenAI-compatible server."""
import asyncio
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Lock, Thread

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams


@dataclass(slots=True)
class TokenEvent:
    token_id: int | None = None
    finished: bool = False
    cancelled: bool = False
    metrics: dict | None = None
    error: str | None = None


@dataclass(slots=True)
class RequestHandle:
    seq_id: int
    events: Queue


class AsyncLLMEngine:

    def __init__(self, model: str, **kwargs):
        self.engine = LLMEngine(model, **kwargs)
        self.tokenizer = self.engine.tokenizer
        self.commands = Queue()
        self.handles = {}
        self._metrics_lock = Lock()
        self._metrics = {
            "requests_total": 0,
            "requests_finished": 0,
            "requests_cancelled": 0,
            "generated_tokens_total": 0,
            "active_requests": 0,
        }
        self.thread = Thread(target=self._run, name="nanovllm-engine", daemon=True)
        self.thread.start()

    async def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        ack = Queue(maxsize=1)
        events = Queue()
        self.commands.put(("add", (prompt, sampling_params, events), ack))
        result = await asyncio.to_thread(ack.get)
        if isinstance(result, Exception):
            raise result
        return RequestHandle(result, events)

    async def next_event(self, handle: RequestHandle):
        return await asyncio.to_thread(handle.events.get)

    def cancel(self, seq_id: int):
        self.commands.put(("cancel", (seq_id,), None))

    async def close(self):
        ack = Queue(maxsize=1)
        self.commands.put(("close", (), ack))
        await asyncio.to_thread(ack.get)
        await asyncio.to_thread(self.thread.join)

    def metrics(self):
        with self._metrics_lock:
            metrics = dict(self._metrics)
        metrics.update(self.engine.metrics())
        return metrics

    def _bump(self, **changes):
        with self._metrics_lock:
            for key, delta in changes.items():
                self._metrics[key] += delta

    def _process_command(self, command, args, ack):
        if command == "add":
            prompt, sampling_params, events = args
            try:
                seq_id = self.engine.add_request(prompt, sampling_params)
                self.handles[seq_id] = events
                self._bump(requests_total=1, active_requests=1)
                ack.put(seq_id)
            except Exception as exc:
                ack.put(exc)
        elif command == "cancel":
            seq_id, = args
            if self.engine.cancel_request(seq_id):
                events = self.handles.pop(seq_id, None)
                if events is not None:
                    events.put(TokenEvent(finished=True, cancelled=True))
                self._bump(requests_cancelled=1, active_requests=-1)
        elif command == "close":
            for seq_id in list(self.handles):
                self.engine.cancel_request(seq_id)
                self.handles[seq_id].put(TokenEvent(finished=True, cancelled=True))
            self.handles.clear()
            self.engine.exit()
            ack.put(True)
            return False
        return True

    def _run(self):
        running = True
        while running:
            if self.engine.is_finished():
                try:
                    item = self.commands.get(timeout=0.1)
                except Empty:
                    continue
                running = self._process_command(*item)
                continue

            while True:
                try:
                    item = self.commands.get_nowait()
                except Empty:
                    break
                running = self._process_command(*item)
                if not running:
                    break
            if not running:
                break

            try:
                self.engine.step()
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                for seq_id, events in list(self.handles.items()):
                    events.put(TokenEvent(finished=True, error=message))
                    self.engine.cancel_request(seq_id)
                self.handles.clear()
                with self._metrics_lock:
                    self._metrics["active_requests"] = 0
                continue

            for seq_id, token_id, finished, metrics in self.engine.last_token_events:
                events = self.handles.get(seq_id)
                if events is None:
                    continue
                events.put(TokenEvent(token_id=token_id, finished=finished, metrics=metrics))
                self._bump(generated_tokens_total=1)
                if finished:
                    self.handles.pop(seq_id, None)
                    self._bump(requests_finished=1, active_requests=-1)
