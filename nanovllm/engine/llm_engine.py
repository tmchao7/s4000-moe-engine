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

# E5 diagnostic: skip the speculative schedule (keep the launch/collect rhythm)
_OVERLAP_NO_SPEC = [os.environ.get("NANOVLLM_OVERLAP_NO_SPEC", "0") == "1"]

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler, ScheduledBatch
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
        # P23 overlap pipeline (one batch in flight; driver-local state)
        self._overlap = bool(getattr(config, "scheduler_overlap", False))
        self._pending = None            # launched, uncollected ScheduledBatch
        self._sched_next = None         # speculative schedule for the next call
        self._pending_launch_time = 0.0
        self.overlap_defers = 0         # speculative passes that fell back
        self.overlap_pruned_rows = 0    # stale rows dropped before launch
        # P24 structured output (xgrammar), compiled lazily on first use
        self._grammar = None            # (xgrammar_mod, TokenizerInfo, GrammarCompiler)
        self._grammar_cache = {}        # (kind, key) -> CompiledGrammar
        self._shm_name = config.shm_name
        if config.kv_cache_dtype == "int8":
            # P19: per-channel scale calibration over a few real prompts
            # (mixed languages/lengths), then frozen for serving.  Saturating
            # garbage during calibration is fine — only activation amax is
            # harvested (hooks; PREFILL forwards only — graph-replayed decode
            # steps bypass Python hooks, and production is eager anyway).
            # begin/freeze go through call() so TP workers calibrate
            # symmetrically.  Exception-safe: a failure tears the engine down
            # instead of leaking the spawned TP worker (which would hang exit).
            try:
                self.model_runner.call("kv_calibrate_begin")
                calib_prompts = [
                    "用一段话解释大模型推理中的 KV cache 与 paged attention。",
                    "Write a Python function that computes moving averages.",
                    "请比较 TCP 与 UDP 的区别，并给出各自适用场景。",
                    "Summarize how continuous batching improves GPU utilization.",
                    "计算 17 × 23 并给出中间步骤，然后用一句话解释竖式乘法。",
                    "Explain why decode is memory-bandwidth bound in one paragraph.",
                    "写一首关于秋天的四行诗。",
                    "List three trade-offs between throughput and latency in serving.",
                ]
                sp = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
                self.generate(calib_prompts, [sp] * len(calib_prompts), use_tqdm=False)
                self.scheduler.block_manager.clear_prefix_cache()
                self.model_runner.call("kv_calibrate_freeze")
            except Exception:
                self.exit()
                raise
            self.step_count = 0
            self.schedule_mode_counts.clear()
            self.prefill_tokens_total = 0
            self.decode_tokens_total = 0
            self.schedule_time_total = 0.0
            self.forward_time_total = 0.0
            self.last_token_events = []
            self.last_step_metrics = None
            self.scheduler.preemptions = 0
            self.scheduler.kv_admission_skips_total = 0
            bm = self.scheduler.block_manager
            bm.prefix_cache_hit_tokens = 0
            bm.allocations = 0
            bm.reclaims = 0
            bm.evictions = 0
            print("[kv-int8] calibration complete; counters reset", flush=True)
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
        if seq.guided_json is not None or seq.guided_regex is not None:
            self._attach_grammar(seq)
        self.scheduler.add(seq)
        return seq.seq_id

    def _attach_grammar(self, seq: Sequence):
        """P24: compile (cached) and attach a stateful GrammarMatcher.  The
        bitmask width is stashed on the runner for prepare_sample (rank 0
        only; matchers never cross the TP pickle boundary)."""
        try:
            import xgrammar as xg
        except ImportError as exc:
            raise ValueError(
                "guided generation requires xgrammar (pip install xgrammar)"
            ) from exc
        if self._grammar is None:
            info = xg.TokenizerInfo.from_huggingface(self.tokenizer)
            self._grammar = (xg, info, xg.GrammarCompiler(info))
        xg, info, compiler = self._grammar
        if seq.guided_json is not None:
            schema = seq.guided_json if isinstance(seq.guided_json, str) \
                else json.dumps(seq.guided_json, sort_keys=True)
            key = ("json", schema)
            build = lambda: compiler.compile_json_schema(schema)
        else:
            key = ("regex", seq.guided_regex)
            build = lambda: compiler.compile_regex(seq.guided_regex)
        compiled = self._grammar_cache.get(key)
        if compiled is None:
            compiled = build()
            self._grammar_cache[key] = compiled
        seq.grammar_matcher = xg.GrammarMatcher(compiled)
        self.model_runner._grammar_words = (info.vocab_size + 31) // 32

    def cancel_request(self, seq_id: int):
        return self.scheduler.cancel(seq_id)

    def set_scheduler_overlap(self, enabled: bool):
        """P23 runtime switch.  Driver-local by design: the pipeline state
        machine lives on the engine thread; TP ranks only ever see
        run_launch/collect commands.  Disabling DRAINS instead of dropping:
        the next steps keep running the overlap loop until the parked
        speculative batch is launched and the in-flight batch collected, and
        only the trailing speculative scheduling stops.  Dropping the parked
        batch would strand the blocks its may_append_at already allocated —
        the serial invariant ``len(block_table) == ceil(len/bs)`` breaks and
        prepare_decode writes KV one block off (P25/F2, silent corruption on
        boundary-crossing rows)."""
        self._overlap = bool(enabled)

    def step(self):
        # Drain routing (P25/F2): with overlap disabled the pipeline may still
        # hold a parked speculative batch or an in-flight launch; _step_overlap
        # retires both before the serial path resumes.
        if self._overlap or self._pending is not None or self._sched_next is not None:
            return self._step_overlap()
        batch = self.scheduler.schedule()
        seqs, mode = batch
        if not seqs:
            # Scheduler found nothing runnable (e.g. KV exhausted, requests
            # waiting for a cancel): report an empty step instead of feeding an
            # empty batch to the model.
            self.last_step_metrics = {
                "mode": "empty", "schedule_s": batch.schedule_s, "forward_s": 0.0,
            }
            return [], 0
        self._pending_launch_time = perf_counter()
        token_ids, logprobs = self.model_runner.call("run", seqs, mode)
        return self._finish_step(batch, token_ids, logprobs)

    def _step_overlap(self):
        """P23: one-batch-in-flight pipeline (SGLang overlap-scheduler shape).

        Per call: collect the in-flight launch (the step's single D2H sync),
        postprocess it, launch the batch speculative-scheduled last call —
        pruning rows for seqs that finished since (pruning BEFORE the launch
        keeps the sampler's row set and RNG stream identical to the serial
        path) — then speculatively schedule the next batch while the GPU runs
        this one.  Speculative scheduling never preempts (stale victims); a
        defer falls back to the serial schedule on the next call, which by
        then runs on exactly the state the serial path would see.

        P25/F2 drain: with ``_overlap`` False the loop still collects the
        in-flight batch and launches the parked speculative batch (its launch-
        time block coverage holds exactly once the collect above appended the
        in-flight rows' tokens); only the trailing speculative schedule is
        gated off.  After at most two drain calls ``_pending`` and
        ``_sched_next`` are both None and the serial step resumes."""
        outputs, num_tokens = [], 0
        collected = False
        if self._pending is not None:
            token_ids, logprobs = self.model_runner.call("collect")
            outputs, num_tokens = self._finish_step(self._pending, token_ids, logprobs)
            self._pending = None
            collected = True
        batch = self._sched_next
        self._sched_next = None
        if batch is not None:
            # The parked batch launches even while draining: dropping it here
            # would strand its pre-allocated blocks (P25/F2).
            # _prune_batch counts each dropped row; do not add a batch delta
            # here or the metric double-counts (review finding).
            batch = self._prune_batch(batch)
        elif self._overlap:
            batch = self.scheduler.schedule()
        if batch is None or batch.mode == "defer" or not batch.sequences:
            if not collected:
                self.last_step_metrics = {
                    "mode": "empty",
                    "schedule_s": batch.schedule_s if batch is not None else 0.0,
                    "forward_s": 0.0,
                }
            return outputs, num_tokens
        self._pending_launch_time = perf_counter()
        self.model_runner.call("run_launch", batch.sequences, batch.mode)
        self._pending = batch
        if not collected:
            # Launch-only call (first step of the pipeline): report the
            # batch now in flight; a collected step's metrics were already
            # set by _finish_step and must not be overwritten.
            self.last_step_metrics = {
                "mode": batch.mode, "schedule_s": batch.schedule_s, "forward_s": 0.0,
            }
        if self._overlap:
            # Trailing speculative schedule — gated off while draining, and
            # also by the E5 diagnostic NANOVLLM_OVERLAP_NO_SPEC=1 (which
            # keeps the launch/collect rhythm but schedules every batch
            # synchronously after collect; scheduling speculatively and
            # dropping the result would strand its block allocations).
            if not _OVERLAP_NO_SPEC[0]:
                next_batch = self.scheduler.schedule(speculative=True,
                                                     inflight=batch.sequences)
                if next_batch.mode == "defer":
                    self.overlap_defers += 1
                self._sched_next = None if (next_batch.mode == "defer"
                                            or not next_batch.sequences) else next_batch
        return outputs, num_tokens

    def _prune_batch(self, batch: ScheduledBatch):
        """Drop stale rows before launching a parked speculative batch.

        Row shape decisions use the batch FROZEN metadata (query_lens,
        prefill_flags); the live values that matter are num_cached_tokens
        (advanced by this step's collect) and is_prefill (cleared by
        postprocess only when the final prefill chunk completes).  The
        trailing speculative schedule rewrites live seq fields for the NEXT
        batch, so live is_prefill must not be used to infer row type.

        Two stale cases:
        1. A chunk-continuing prefill row whose earlier chunks completed in
           the in-flight batch: clamp to the true remaining tokens.
        2. A prefill row whose final chunk completed in the in-flight batch
           (postprocess cleared live is_prefill): DROP it.  Keeping it
           produced a 1-token prefill row with num_cached == prompt_len,
           whose block_table[num_cached // block_size] read is out of range
           when the prompt length is an exact multiple of block_size
           (P25/F2 follow-up crash).  The seq decodes from the next schedule.
        A preempted seq is re-prefilled from 0 with num_tokens >
        num_prompt_tokens, so it may legitimately cross num_prompt_tokens
        mid-way; the live is_prefill flag is the completion signal, not the
        prompt length.
        Decode rows are never clamped: their 1 token is generated, not read
        from token_ids."""
        flags = batch.prefill_flags
        assert len(flags) == len(batch.sequences) == len(batch.query_lens), (
            "frozen row metadata mismatch: "
            f"flags={len(flags)} seqs={len(batch.sequences)} "
            f"query_lens={len(batch.query_lens)}"
        )
        kept = []
        kept_flags = []
        for idx, (seq, q) in enumerate(zip(batch.sequences, batch.query_lens)):
            was_prefill = flags[idx]
            if seq.is_finished:
                self.overlap_pruned_rows += 1
                continue
            if was_prefill:
                # postprocess clears the live is_prefill flag only when the
                # final prefill chunk completes.  Do NOT use
                # num_cached >= num_prompt_tokens here: a preempted seq is
                # re-prefilled from 0 with num_tokens > num_prompt_tokens, so
                # a legitimate mid-way chunk can cross the prompt length and
                # would be wrongly dropped (C1 regression found in review).
                if not seq.is_prefill:
                    self.overlap_pruned_rows += 1
                    seq.num_scheduled_tokens = 0
                    continue
                q = min(q, seq.num_tokens - seq.num_cached_tokens)
            if q <= 0:
                # stale prefill row whose chunks all completed in between;
                # the seq decodes from the next schedule on (recorded
                # divergence, same class as the arrival-timing shift)
                self.overlap_pruned_rows += 1
                seq.num_scheduled_tokens = 0
                seq.is_prefill = False
                continue
            seq.num_scheduled_tokens = q
            seq.is_prefill = was_prefill
            kept.append(seq)
            kept_flags.append(was_prefill)
        if not kept:
            return None
        has_dec = any(not flag for flag in kept_flags)
        has_pre = any(kept_flags)
        mode = "mixed" if (has_dec and has_pre) else ("decode" if has_dec else "prefill")
        return ScheduledBatch(
            sequences=kept,
            query_lens=[seq.num_scheduled_tokens for seq in kept],
            context_lens=[seq.num_cached_tokens + seq.num_scheduled_tokens
                          for seq in kept],
            num_scheduled_tokens=sum(seq.num_scheduled_tokens for seq in kept),
            mode=mode,
            schedule_s=batch.schedule_s,
            prefill_flags=kept_flags,
        )

    def _finish_step(self, batch: ScheduledBatch, token_ids, logprobs):
        """Shared tail of a completed step (serial and overlap): postprocess,
        counters, metrics, JSONL record, finished-output assembly."""
        seqs, mode = batch.sequences, batch.mode
        self.schedule_mode_counts[mode] += 1
        flags = batch.prefill_flags
        query_lens = batch.query_lens
        assert len(flags) == len(query_lens) == len(seqs), (
            "frozen row metadata mismatch: "
            f"flags={len(flags)} query_lens={len(query_lens)} seqs={len(seqs)}"
        )
        # Frozen row metadata, not live seq state: the overlap pipeline has
        # already rewritten the live fields for the next batch by the time
        # this batch is collected (C2 accounting fix).
        prefill_tokens = sum(q for q, flag in zip(query_lens, flags) if flag)
        decode_tokens = sum(q for q, flag in zip(query_lens, flags) if not flag)
        num_tokens = batch.num_scheduled_tokens if mode != "decode" else -len(seqs)
        forward_elapsed = perf_counter() - self._pending_launch_time
        self.last_token_events = self.scheduler.postprocess(
            seqs, token_ids, mode, logprobs, query_lens, flags
        )
        self.step_count += 1
        self.prefill_tokens_total += prefill_tokens
        self.decode_tokens_total += decode_tokens
        self.schedule_time_total += batch.schedule_s
        self.forward_time_total += forward_elapsed
        self.last_step_metrics = {
            "mode": mode,
            "schedule_s": batch.schedule_s,
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
                "schedule_s": batch.schedule_s,
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
        result["engine_overlap_defers_total"] = self.overlap_defers
        result["engine_overlap_pruned_rows_total"] = self.overlap_pruned_rows
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
