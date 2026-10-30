import atexit
import json
import os
import socket
import sys
import threading
import uuid
from collections import Counter
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


def _get_distributed_init_method() -> str:
    """Reserve an ephemeral localhost port for this engine instance.

    The socket is closed before torch.distributed binds it, so there is a very
    small race, but unlike the old fixed port this permits sequential tests and
    independent server processes without guaranteed collisions.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        if not config.distributed_init_method:
            config.distributed_init_method = _get_distributed_init_method()
        if not config.shm_name:
            config.shm_name = f"nanovllm_{uuid.uuid4().hex}"
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.last_token_events = []
        self.schedule_mode_counts = Counter()
        self.step_count = 0
        self.prefill_tokens_total = 0
        self.decode_tokens_total = 0
        self.schedule_time_total = 0.0
        self.forward_time_total = 0.0
        self.metrics_path = config.metrics_path
        self.last_step_metrics = None
        self._shm_name = config.shm_name
        atexit.register(self.exit)

    def _force_exit(self):
        """Last-resort teardown when a rank is wedged in exit (I12).

        Runs on the watchdog thread: kill spawned ranks, best-effort unlink the
        shared-memory segment, then leave without running further atexit hooks.
        """
        print(
            "[engine] exit watchdog fired: teardown still blocked; killing ranks "
            "and forcing process exit (I12)",
            file=sys.stderr, flush=True,
        )
        for p in getattr(self, "ps", []):
            try:
                p.kill()
            except Exception:
                pass
        name = getattr(self, "_shm_name", "")
        if name:
            try:
                os.unlink(f"/dev/shm/{name}")
            except OSError:
                pass
        os._exit(0)

    def exit(self):
        if not hasattr(self, "model_runner"):
            return
        # I12: a wedged rank (DistBackendError, stuck barrier) used to hang the
        # main process in this path and needed a manual kill by PID. A daemon
        # timer now bounds the teardown; NANOVLLM_EXIT_TIMEOUT_S=0 disables it.
        timeout = float(os.environ.get("NANOVLLM_EXIT_TIMEOUT_S", "60"))
        watchdog = None
        if timeout > 0:
            watchdog = threading.Timer(timeout, self._force_exit)
            watchdog.daemon = True
            watchdog.start()
        try:
            self.model_runner.call("exit")
            del self.model_runner
            for p in self.ps:
                p.join()
        finally:
            if watchdog is not None:
                watchdog.cancel()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        total = len(prompt) + sampling_params.max_tokens
        config = self.model_runner.config
        kv_token_capacity = config.num_kvcache_blocks * self.model_runner.block_size
        if total > config.max_model_len:
            raise ValueError(
                f"prompt({len(prompt)}) + max_tokens({sampling_params.max_tokens}) "
                f"exceeds max_model_len({config.max_model_len})"
            )
        if total > kv_token_capacity:
            # Recompute-only preemption can never admit a sequence that does not
            # fit the KV pool; reject instead of queueing it forever.
            raise OverflowError(
                f"request needs {total} tokens but KV capacity is "
                f"{kv_token_capacity} tokens"
            )
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id

    def cancel_request(self, seq_id: int):
        return self.scheduler.cancel(seq_id)

    def step(self):
        schedule_start = perf_counter()
        batch = self.scheduler.schedule()
        schedule_elapsed = perf_counter() - schedule_start
        seqs, mode = batch
        if not seqs:
            # Scheduler found nothing runnable (e.g. KV exhausted, requests
            # waiting for a cancel): report an empty step instead of feeding an
            # empty batch to the model.
            self.last_step_metrics = {
                "mode": "empty", "schedule_s": schedule_elapsed, "forward_s": 0.0,
            }
            return [], 0
        self.schedule_mode_counts[mode] += 1
        prefill_tokens = sum(
            seq.num_scheduled_tokens for seq in seqs if seq.is_prefill
        )
        decode_tokens = sum(
            seq.num_scheduled_tokens for seq in seqs if not seq.is_prefill
        )
        num_tokens = batch.num_scheduled_tokens if mode != "decode" else -len(seqs)
        forward_start = perf_counter()
        token_ids = self.model_runner.call("run", seqs, mode)
        forward_elapsed = perf_counter() - forward_start
        self.last_token_events = self.scheduler.postprocess(seqs, token_ids, mode)
        self.step_count += 1
        self.prefill_tokens_total += prefill_tokens
        self.decode_tokens_total += decode_tokens
        self.schedule_time_total += schedule_elapsed
        self.forward_time_total += forward_elapsed
        self.last_step_metrics = {
            "mode": mode,
            "schedule_s": schedule_elapsed,
            "forward_s": forward_elapsed,
        }
        if self.metrics_path:
            record = {
                "kind": "step",
                "step": self.step_count,
                "mode": mode,
                "effective_batch": len(seqs),
                "prefill_tokens": prefill_tokens,
                "decode_tokens": decode_tokens,
                "scheduled_tokens": batch.num_scheduled_tokens,
                "query_lens": batch.query_lens,
                "context_lens": batch.context_lens,
                "schedule_s": schedule_elapsed,
                "forward_s": forward_elapsed,
            }
            with open(self.metrics_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        outputs = [
            (seq.seq_id, seq.completion_token_ids, seq.metrics())
            for seq in seqs if seq.is_finished
        ]
        return outputs, num_tokens

    def metrics(self):
        result = {
            "engine_steps_total": self.step_count,
            "engine_prefill_tokens_total": self.prefill_tokens_total,
            "engine_decode_tokens_total": self.decode_tokens_total,
            "engine_schedule_seconds_total": self.schedule_time_total,
            "engine_forward_seconds_total": self.forward_time_total,
        }
        for mode, count in self.schedule_mode_counts.items():
            result[f"engine_{mode}_steps_total"] = count
        result.update(self.scheduler.metrics())
        return result

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids, metrics in output:
                outputs[seq_id] = (token_ids, metrics)
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids, "metrics": metrics}
            for token_ids, metrics in outputs
        ]
        return outputs
