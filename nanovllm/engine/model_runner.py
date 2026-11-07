import gc
import os
import pickle
import time
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.profiling import DeviceProfiler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.compat import DEVICE, IS_MUSA, DIST_BACKEND, accelerator


def _model_cls(hf_config):
    arch = (getattr(hf_config, "architectures", None) or [""])[0]
    if "Moe" in arch:
        return Qwen3MoeForCausalLM
    return Qwen3ForCausalLM


_TRACE_INIT = os.environ.get("NANOVLLM_TRACE_INIT") == "1"

# Runtime-switchable for interleaved A/B of the host-sync removals (sampler
# all_greedy + prefill cu_seqlens lists).  Defaults on; set
# NANOVLLM_HOST_SYNC_FIX=0 or call set_host_sync_fix(False) to restore the
# per-step device->host syncs.
_SYNC_FIX = [os.environ.get("NANOVLLM_HOST_SYNC_FIX", "1") == "1"]


def set_host_sync_fix(enabled: bool) -> None:
    _SYNC_FIX[0] = bool(enabled)


# Mixed steps used to run the whole batch through the prefill path, so decode
# rows lost paged attention and bmm_dense (I23).  The split runs each part
# through its own prepare/forward; runtime-switchable for A/B.
_MIXED_SPLIT = [os.environ.get("NANOVLLM_MIXED_SPLIT", "1") == "1"]

# I29 segmented-graph prototype (P10): capture per-layer segment graphs with
# ZERO collectives inside (the two per-layer all_reduces run eagerly between
# segment replays).  TP2 whole-graph replay wedges (I29); TP1 graphs (no
# in-graph collectives) never do — this tests whether removing in-graph MCCL
# from the ENGINE replay path dodges the wedge.  Default off.
_SEGMENT_GRAPH = [os.environ.get("NANOVLLM_SEGMENT_GRAPH", "0") == "1"]
# P11: capture-time instrumentation for the two-engine illegal-access
# bisect (P2) and capture-flake slow-start (P1).  Both default inert.
_SEG_DEBUG = os.environ.get("NANOVLLM_SEG_DEBUG", "0") == "1"
_SEG_SLOW = float(os.environ.get("NANOVLLM_SEG_SLOW", "0"))
# I33 bucket-padding controls, both default inert (P12 online A/B verdict kept).
#   NANOVLLM_SEGMENT_GRAPH_BUCKETS=dense -> vLLM-style ladder [1,2,4]+step-8.
#     Online concurrency 14-30 sat between the sparse default buckets and paid
#     up to 1.88x padding compute; dense caps that at 1.41x.  Graph count grows
#     to 2*layers*len(buckets) (1056/rank at max_num_seqs=64); the MUSA driver
#     caps live graph executables around ~2048 (torchada), i.e. ~21 buckets at
#     48 layers x 2 segments.
#   NANOVLLM_SEGMENT_GRAPH_PAD_RATIO=r   -> when cap/bs > 1+r, run the decode
#     step eager instead of replaying a padded graph (SGLang
#     --disable-cuda-graph-padding semantics; default 0 keeps replay-always).
_SEG_BUCKETS = os.environ.get("NANOVLLM_SEGMENT_GRAPH_BUCKETS", "default")
_SEG_PAD_RATIO = float(os.environ.get("NANOVLLM_SEGMENT_GRAPH_PAD_RATIO", "0"))


def set_mixed_split(enabled: bool) -> None:
    _MIXED_SPLIT[0] = bool(enabled)


def _trace_init(rank, stage, started=None):
    """Print init milestones when NANOVLLM_TRACE_INIT=1.

    Startup is silent from ``load_model`` through ``allocate_kv_cache``; when it
    hangs there is no way to tell which stage is stuck.  Both ranks write to the
    inherited stdout, so one server log shows the interleaving.
    """
    if not _TRACE_INIT:
        return
    elapsed = "" if started is None else f" dt={time.perf_counter() - started:.2f}s"
    print(f"[init] rank{rank} {stage} t={time.time():.3f}{elapsed}", flush=True)


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        self.device = DEVICE
        self.acc = accelerator()
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        _trace_init(rank, "start")
        dist.init_process_group(
            DIST_BACKEND,
            config.distributed_init_method,
            world_size=self.world_size,
            rank=rank,
        )
        _trace_init(rank, "dist_ready")
        self.acc.set_device(rank)
        default_dtype = torch.get_default_dtype()
        model_dtype = hf_config.dtype
        if isinstance(model_dtype, str):
            model_dtype = getattr(torch, model_dtype)
        self.model_dtype = model_dtype
        torch.set_default_dtype(model_dtype)
        torch.set_default_device(self.device)
        model_cls = _model_cls(hf_config)
        if model_cls is Qwen3MoeForCausalLM:
            self.model = model_cls(hf_config, config)
        else:
            self.model = model_cls(hf_config)
        self.profiler = DeviceProfiler(self.acc)
        attention_layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "profiler"):
                module.profiler = self.profiler
            if hasattr(module, "attention_backend"):
                module.attention_backend = config.attention_backend
                module.layer_id = attention_layer_id
                attention_layer_id += 1
        _trace_init(rank, "model_built")
        stage_started = time.perf_counter()
        load_model(self.model, config.model)
        _trace_init(rank, "weights_loaded", stage_started)
        self.sampler = Sampler()
        stage_started = time.perf_counter()
        self.warmup_model()
        _trace_init(rank, "warmup_done", stage_started)
        stage_started = time.perf_counter()
        self.allocate_kv_cache()
        _trace_init(rank, "kv_allocated", stage_started)
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        if config.enable_profiling:
            self.profiler.start()

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name=config.shm_name, create=True, size=2**20)
                dist.barrier()
                _trace_init(rank, "handshake_done")
            else:
                dist.barrier()
                self.shm = SharedMemory(name=config.shm_name)
                _trace_init(rank, "handshake_done")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            # Whole-graph mode sets graphs/graph_pool; segment mode sets
            # seg_a/seg_b/seg_bufs/graph_pool — delete whichever exist.
            for attr in ("graphs", "seg_a", "seg_b", "seg_bufs", "graph_pool"):
                if hasattr(self, attr):
                    delattr(self, attr)
        self.acc.synchronize()
        # Drop the device tensors we own and release the caching allocator's
        # reserved segments.  k_cache/v_cache are views into self.kv_cache held
        # by model modules, so deleting self.model releases both.  Without this,
        # back-to-back LLM builds in one process (p2/p6 run six configs
        # sequentially) see each other's device-level "used" in the KV budget
        # heuristic and the second build asserts "no KV cache memory".
        for attr in ("model", "kv_cache"):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        self.acc.empty_cache()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        result = method(*args)
        if self.world_size > 1 and method_name != "exit":
            # The shm command channel is a single buffer with no ack: rank 0
            # writes the next command as soon as it returns from the local call,
            # so rank 1 can still be unpickling the previous one.  This showed
            # up as "_pickle.UnpicklingError: pickle data was truncated" when
            # the A/B harness called set_enforce_eager / set_moe_backend /
            # set_attention_backend back to back, and as rank 1 executing a
            # different step than rank 0 (the historical b32 non-reproducibility
            # and b64 "MUSA unknown error" crashes).  One barrier per command
            # (~0.08 ms) makes the channel safe; ``exit`` is excluded because
            # it destroys the process group itself.
            dist.barrier()
        return result

    def set_moe_backend(self, backend: str, decode_backend: str | None = None):
        """Set the MoE backend; ``decode_backend`` defaults to the same one."""
        assert backend in (
            "loop", "grouped_mm", "fused_grouped_mm", "bmm_dense", "vllm_fused"
        )
        if decode_backend is not None:
            assert decode_backend in (
                "loop", "grouped_mm", "fused_grouped_mm", "bmm_dense", "vllm_fused"
            )
        for module in self.model.modules():
            if hasattr(module, "moe_backend"):
                module.moe_backend = backend
                module.decode_moe_backend = decode_backend or backend

    def set_attention_backend(self, backend: str):
        assert backend in ("sdpa", "triton_paged")
        self._set_attention_backend_local(backend)

    def _set_attention_backend_local(self, backend: str):
        """Set the attention backend in this rank without using the SHM channel.

        ``probe_attention_backends`` invokes this helper twice inside one
        already-broadcast command.  Calling ``set_attention_backend`` there
        would recursively publish another command and desynchronise TP ranks.
        """
        for module in self.model.modules():
            if hasattr(module, "attention_backend"):
                module.attention_backend = backend

    def _snapshot_kv_slots(self, slot_mapping: torch.Tensor) -> torch.Tensor:
        """Clone the KV entries written by one decode step on this TP rank."""
        slots = slot_mapping.to(torch.long)
        flat_kv = self.kv_cache.view(
            2,
            self.kv_cache.size(1),
            -1,
            self.kv_cache.size(-2),
            self.kv_cache.size(-1),
        )
        return flat_kv[:, :, slots, :, :].clone()

    def _restore_kv_slots(self, slot_mapping: torch.Tensor, snapshot: torch.Tensor):
        """Restore a matching ``_snapshot_kv_slots`` result in place."""
        slots = slot_mapping.to(torch.long)
        flat_kv = self.kv_cache.view(
            2,
            self.kv_cache.size(1),
            -1,
            self.kv_cache.size(-2),
            self.kv_cache.size(-1),
        )
        flat_kv[:, :, slots, :, :] = snapshot

    @staticmethod
    def _logit_probe_summary(
        baseline_logits: torch.Tensor,
        candidate_logits: torch.Tensor,
        baseline_layers: dict[int, torch.Tensor],
        candidate_layers: dict[int, torch.Tensor],
    ) -> dict:
        """Return compact, JSON-safe diagnostics for an SDPA/Triton decode pair.

        This runs only on rank 0, after ``ParallelLMHead`` has gathered the
        full vocabulary.  Keeping all arithmetic here avoids writing or
        sending multi-megabyte logits tensors outside the diagnostic process.
        """
        baseline = baseline_logits.float()
        candidate = candidate_logits.float()
        delta = (candidate - baseline).abs()
        finite = bool(torch.isfinite(delta).all().item())
        baseline_values, baseline_ids = torch.topk(baseline, k=2, dim=-1)
        candidate_values, candidate_ids = torch.topk(candidate, k=2, dim=-1)
        request_max = delta.max(dim=-1).values
        request_mean = delta.mean(dim=-1)

        requests = []
        for index in range(baseline.size(0)):
            baseline_top1 = int(baseline_ids[index, 0].item())
            candidate_top1 = int(candidate_ids[index, 0].item())
            requests.append({
                "request_index": index,
                "top1_equal": baseline_top1 == candidate_top1,
                "baseline_top1": baseline_top1,
                "candidate_top1": candidate_top1,
                "baseline_margin": float((baseline_values[index, 0] - baseline_values[index, 1]).item()),
                "candidate_margin": float((candidate_values[index, 0] - candidate_values[index, 1]).item()),
                "candidate_delta_at_baseline_top1": float(
                    (candidate[index, baseline_top1] - baseline[index, baseline_top1]).item()
                ),
                "baseline_delta_at_candidate_top1": float(
                    (candidate[index, candidate_top1] - baseline[index, candidate_top1]).item()
                ),
                "max_abs": float(request_max[index].item()) if finite else None,
                "mean_abs": float(request_mean[index].item()) if finite else None,
            })

        layers = []
        for layer_id in sorted(set(baseline_layers) | set(candidate_layers)):
            baseline_output = baseline_layers.get(layer_id)
            candidate_output = candidate_layers.get(layer_id)
            if baseline_output is None or candidate_output is None:
                layers.append({"layer_id": layer_id, "missing": True})
                continue
            layer_delta = (candidate_output.float() - baseline_output.float()).abs()
            layer_finite = bool(torch.isfinite(layer_delta).all().item())
            layers.append({
                "layer_id": layer_id,
                "finite": layer_finite,
                "max_abs": float(layer_delta.max().item()) if layer_finite else None,
                "mean_abs": float(layer_delta.mean().item()) if layer_finite else None,
            })

        return {
            "finite": finite,
            "logits": {
                "max_abs": float(delta.max().item()) if finite else None,
                "mean_abs": float(delta.mean().item()) if finite else None,
                "top1_mismatch_requests": [
                    row["request_index"] for row in requests if not row["top1_equal"]
                ],
            },
            "requests": requests,
            "layers": layers,
        }

    @torch.inference_mode()
    def probe_attention_backends(self, seqs: list[Sequence]):
        """Compare SDPA and paged Triton on exactly one committed decode state.

        The method is diagnostic-only and is called through ``call`` so both TP
        ranks execute the same two forwards.  SDPA is the committed trajectory:
        after measuring Triton we restore the SDPA KV writes before returning
        its sampled token ids.  This makes each later comparison teacher-forced
        on the same baseline history instead of amplifying an earlier argmax
        difference.
        """
        assert self.enforce_eager, "attention probe requires eager decode"
        input_ids, positions = self.prepare_decode(seqs)
        temperatures, all_greedy = self.prepare_sample(seqs) if self.rank == 0 else (None, True)
        context = get_context()
        slot_mapping = context.slot_mapping
        original_slots = self._snapshot_kv_slots(slot_mapping)
        baseline_slots = None
        previous_backend = next(
            module.attention_backend
            for module in self.model.modules()
            if hasattr(module, "attention_backend")
        )
        baseline_layers: dict[int, torch.Tensor] = {}
        candidate_layers: dict[int, torch.Tensor] = {}
        hook_handles = []
        succeeded = False

        def install_hooks(destination: dict[int, torch.Tensor]):
            if self.rank != 0:
                return []

            def capture(layer_id: int):
                def hook(_module, _inputs, output):
                    destination[layer_id] = output.detach().clone()
                return hook

            return [
                module.register_forward_hook(capture(module.layer_id))
                for module in self.model.modules()
                if hasattr(module, "attention_backend") and hasattr(module, "layer_id")
            ]

        try:
            self._set_attention_backend_local("sdpa")
            hook_handles = install_hooks(baseline_layers)
            baseline_logits = self.run_model(input_ids, positions, "decode")
            for handle in hook_handles:
                handle.remove()
            hook_handles = []
            baseline_slots = self._snapshot_kv_slots(slot_mapping)

            self._restore_kv_slots(slot_mapping, original_slots)
            self._set_attention_backend_local("triton_paged")
            hook_handles = install_hooks(candidate_layers)
            candidate_logits = self.run_model(input_ids, positions, "decode")
            for handle in hook_handles:
                handle.remove()
            hook_handles = []

            diagnostics = None
            token_ids = None
            if self.rank == 0:
                diagnostics = self._logit_probe_summary(
                    baseline_logits, candidate_logits, baseline_layers, candidate_layers
                )
                token_ids = self.sampler(baseline_logits, temperatures, all_greedy).tolist()

            self._restore_kv_slots(slot_mapping, baseline_slots)
            succeeded = True
            return {"token_ids": token_ids, "diagnostics": diagnostics}
        finally:
            for handle in hook_handles:
                handle.remove()
            if not succeeded:
                self._restore_kv_slots(slot_mapping, original_slots)
            self._set_attention_backend_local(previous_backend)
            reset_context()

    def set_enforce_eager(self, value: bool):
        """Switch between graph replay and eager forward without reloading.

        The graphs captured at init stay alive, so an eager-vs-graph A/B can
        run inside one process/one weight load.
        """
        self.enforce_eager = bool(value)
        return self.enforce_eager

    def graph_report(self):
        return {
            "rank": self.rank,
            "enabled": not self.enforce_eager,
            "buckets": getattr(self, "graph_bs", None),
            "decode_moe_backend": getattr(self, "captured_moe_backend", None),
            "allocated_delta_gib": getattr(self, "graph_memory_delta_gib", None),
            "reserved_delta_gib": getattr(self, "graph_memory_reserved_delta_gib", None),
            "memory": self.memory_report(),
        }

    def memory_report(self):
        gib = 2 ** 30

        def _get(name):
            fn = getattr(self.acc, name, None)
            return fn() / gib if fn is not None else None

        return {
            "rank": self.rank,
            "allocated_gib": _get("memory_allocated"),
            "reserved_gib": _get("memory_reserved"),
            "peak_gib": _get("max_memory_allocated"),
        }

    def start_profiling(self):
        self.profiler.start(reset=True)

    def profile_snapshot(self):
        local = {"rank": self.rank, **self.profiler.snapshot(reset=True)}
        if self.world_size == 1:
            return [local]
        gathered = [None for _ in range(self.world_size)] if self.rank == 0 else None
        dist.gather_object(local, gathered, dst=0)
        return gathered if self.rank == 0 else None

    def stop_profiling(self):
        self.profiler.stop()

    def warmup_model(self):
        self.acc.empty_cache()
        self.acc.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, "prefill")
        self.acc.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = self.acc.mem_get_info()
        used = total - free
        peak = self.acc.memory_stats()["allocated_bytes.all.peak"]
        current = self.acc.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * self.model_dtype.itemsize
        # An explicit positive block count wins over the utilisation heuristic:
        # graph capture needs contiguous headroom on top of the KV cache, and the
        # heuristic deliberately fills the device (capture then OOMs, e.g. 384 MiB
        # short with 44.29 GiB already allocated).
        explicit_blocks = config.num_kvcache_blocks > 0
        if not explicit_blocks:
            config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0, (
            f"no KV cache memory: total={total / 2**30:.2f} GiB used={used / 2**30:.2f} GiB "
            f"peak={peak / 2**30:.2f} GiB utilization={config.gpu_memory_utilization}"
        )
        if self.world_size > 1:
            # Block tables come from rank 0 only; a rank whose KV tensor is
            # smaller than the largest issued block id indexes out of bounds
            # (seen 730 vs 728 on 30B TP2).  Unify on the minimum.
            blocks_t = torch.tensor(config.num_kvcache_blocks, dtype=torch.int64, device=self.device)
            dist.all_reduce(blocks_t, op=dist.ReduceOp.MIN)
            config.num_kvcache_blocks = int(blocks_t.item())
        print(
            f"[kvcache] rank{self.rank} blocks={config.num_kvcache_blocks} "
            f"block_bytes={block_bytes} kv={config.num_kvcache_blocks * block_bytes / 2**30:.2f} GiB "
            f"explicit={explicit_blocks} total={total / 2**30:.2f} GiB used={used / 2**30:.2f} GiB "
            f"peak={peak / 2**30:.2f} GiB current={current / 2**30:.2f} GiB free={free / 2**30:.2f} GiB",
            flush=True,
        )
        # One extra block the BlockManager never sees: graph replay points the
        # padding rows' slot_mapping and block_tables at it, so the whole-bucket
        # replay cannot overwrite KV slot 0 of a live request.
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks + 1, self.block_size, num_kv_heads, head_dim)
        self.scratch_block_id = config.num_kvcache_blocks
        self.scratch_slot = config.num_kvcache_blocks * self.block_size
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            if seq.is_prefill:
                input_ids.extend(seq[start:end])
            else:
                assert seqlen_q == 1
                input_ids.append(seq.last_token)
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        cu_q_list, cu_k_list = list(cu_seqlens_q), list(cu_seqlens_k)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables,
                    cu_seqlens_q_list=cu_q_list if _SYNC_FIX[0] else None,
                    cu_seqlens_k_list=cu_k_list if _SYNC_FIX[0] else None)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        max_context_len = max(context_lens)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            max_context_len=max_context_len,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures_list = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures_list, dtype=torch.float32, pin_memory=True).to(self.device, non_blocking=True)
        # Host-side greedy decision: replaces the sampler's two bool() device
        # reads (2 host syncs per step).
        all_greedy = all(value < 1e-3 for value in temperatures_list) if _SYNC_FIX[0] else None
        return temperatures, all_greedy

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, mode: str):
        if (mode == "decode" and not self.enforce_eager
                and _SEGMENT_GRAPH[0] and getattr(self, "seg_a", None)
                and input_ids.size(0) <= self.graph_bs[-1]):
            return self._run_model_segment(input_ids, positions)
        if mode != "decode" or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            # Padding rows (``bs:`` up to the bucket) still execute inside the
            # graph.  Their slot_mapping points at the reserved scratch block,
            # never at slot 0: the captured index_put writes every row of the
            # bucket, and a 0 slot overwrites the first KV slot of whichever
            # request owns block 0 -- the observed replay drift.  context_lens=1
            # keeps those rows on a valid one-token attention instead of a 0/0
            # division, and their block_tables point at the scratch block so
            # they read only their own just-written slot, never a live block.
            graph_vars["slot_mapping"].fill_(self.scratch_slot)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].fill_(1)
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"].fill_(self.scratch_block_id)
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def _run_mixed(self, seqs: list[Sequence], temperatures, all_greedy):
        """Split a mixed step so decode rows keep paged attention + bmm_dense.

        Each part runs through its own prepare_* / run_model (the global
        context only holds one batch at a time), then tokens are re-assembled
        in the scheduler's seqs order for postprocess.
        """
        prefill_idx = [i for i, seq in enumerate(seqs) if seq.is_prefill]
        decode_idx = [i for i, seq in enumerate(seqs) if not seq.is_prefill]
        token_ids: list[int | None] = [None] * len(seqs)
        if prefill_idx:
            input_ids, positions = self.prepare_prefill([seqs[i] for i in prefill_idx])
            logits = self.run_model(input_ids, positions, "prefill")
            if self.rank == 0:
                sampled = self.sampler(
                    logits, temperatures[prefill_idx], all_greedy
                ).tolist()
                for row, index in enumerate(prefill_idx):
                    token_ids[index] = sampled[row]
        if decode_idx:
            input_ids, positions = self.prepare_decode([seqs[i] for i in decode_idx])
            logits = self.run_model(input_ids, positions, "decode")
            if self.rank == 0:
                sampled = self.sampler(
                    logits, temperatures[decode_idx], all_greedy
                ).tolist()
                for row, index in enumerate(decode_idx):
                    token_ids[index] = sampled[row]
        reset_context()
        return token_ids

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], mode: str) -> list[int]:
        with self.profiler.range("runner.prepare", mode=mode):
            temperatures, all_greedy = self.prepare_sample(seqs) if self.rank == 0 else (None, True)
            if mode == "mixed" and _MIXED_SPLIT[0]:
                return self._run_mixed(seqs, temperatures, all_greedy)
            input_ids, positions = self.prepare_decode(seqs) if mode == "decode" else self.prepare_prefill(seqs)
        with self.profiler.range("runner.forward", mode=mode):
            logits = self.run_model(input_ids, positions, mode)
        with self.profiler.range("runner.sample", mode=mode):
            token_ids = self.sampler(logits, temperatures, all_greedy).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    def _segment_layers(self):
        return self.model.model.layers

    def _set_segment_defer(self, value: bool):
        for module in self._defer_modules:
            module.defer_reduce = value

    @torch.inference_mode()
    def capture_segment_graphs(self):
        """I29 segmented-graph prototype (P10): per-layer segment graphs with
        ZERO in-graph collectives.  Segment A = ln1 -> qkv/rope/attn ->
        o_proj(partial); segment B = post_ln -> mlp/moe(partial).  The two
        per-layer all_reduces run eagerly between replays in
        _run_model_segment, so the captured graphs never touch MCCL — the
        TP1-style shape that has never wedged (I29)."""
        dbg = (lambda msg: print(f"[graph-seg-dbg] rank{self.rank} {msg}", flush=True)) if _SEG_DEBUG else (lambda msg: None)
        dbg("entry")
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.arange(max_bs, dtype=torch.int32)
        context_lens = torch.full(
            (max_bs,), min(self.block_size, config.max_model_len), dtype=torch.int32
        )
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        if _SEG_BUCKETS == "dense":
            self.graph_bs = [1, 2, 4] + list(range(8, max_bs + 1, 8))
        else:
            self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        if self.graph_bs[-1] != max_bs:
            self.graph_bs.append(max_bs)
        self.graph_vars = dict(
            positions=positions, slot_mapping=slot_mapping,
            context_lens=context_lens, block_tables=block_tables,
        )
        dbg("graph_vars ready")
        layers = self._segment_layers()
        self._defer_modules = []
        for layer in layers:
            self._defer_modules.append(layer.self_attn.o_proj)
            self._defer_modules.append(layer.mlp)          # MoE combine AR
            down = getattr(layer.mlp, "down_proj", None)   # dense MLP AR
            if down is not None:
                self._defer_modules.append(down)
        dtype = next(self.model.parameters()).dtype
        eager_backend = config.moe_backend
        capture_backend = config.decode_moe_backend or eager_backend
        if capture_backend != eager_backend:
            self.set_moe_backend(capture_backend)
        self.captured_moe_backend = capture_backend
        self._set_segment_defer(True)
        dbg("defer on")
        self.seg_a = {li: {} for li in range(len(layers))}
        self.seg_b = {li: {} for li in range(len(layers))}
        self.seg_bufs = {}
        pool = None
        Graph = self.acc.MUSAGraph
        graph_ctx = self.acc.graph
        for bs in self.graph_bs:
            hidden = hf_config.hidden_size
            self.seg_bufs[bs] = dict(
                ping0=torch.zeros(bs, hidden, dtype=dtype, device=self.device),
                ping1=torch.zeros(bs, hidden, dtype=dtype, device=self.device),
                h_attn=torch.zeros(bs, hidden, dtype=dtype, device=self.device),
                r=torch.zeros(bs, hidden, dtype=dtype, device=self.device),
            )
            bufs = self.seg_bufs[bs]
            dbg(f"bucket={bs} bufs allocated")
            if self.world_size > 1:
                dbg(f"bucket={bs} barrier 1")
                dist.barrier()
                dbg(f"bucket={bs} barrier 1 done")
            dbg(f"bucket={bs} warmup start")
            # Warmup every module once per bucket (collectives deferred, values
            # garbage, launches/JIT real), mirroring the whole-graph warmup;
            # same degenerate-but-valid decode context as the engine capture.
            set_context(
                False, slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs], block_tables=block_tables[:bs],
            )
            self.model(
                torch.zeros(bs, dtype=torch.int64, device=self.device),
                positions[:bs],
            )
            dbg(f"bucket={bs} warmup launches done")
            if self.world_size > 1:
                dist.barrier()
                self.acc.synchronize()
            dbg(f"bucket={bs} warmup synced")
            print(f"[graph-seg] rank{self.rank} bucket={bs} warmup done "
                  f"({len(layers)} layers)", flush=True)
            for li, layer in enumerate(layers):
                read = bufs["ping0"] if li % 2 == 0 else bufs["ping1"]
                write = bufs["ping1"] if li % 2 == 0 else bufs["ping0"]
                h_attn, r_buf = bufs["h_attn"], bufs["r"]
                pos_view = positions[:bs]
                if li == 0:
                    def seg_a_body(layer=layer, read=read, h_attn=h_attn,
                                   r_buf=r_buf, pos_view=pos_view):
                        hn = layer.input_layernorm(read)
                        r_buf.copy_(read)
                        h_attn.copy_(layer.self_attn(pos_view, hn))
                else:
                    def seg_a_body(layer=layer, read=read, h_attn=h_attn,
                                   r_buf=r_buf, pos_view=pos_view):
                        hn, r_out = layer.input_layernorm(read, r_buf)
                        h_attn.copy_(layer.self_attn(pos_view, hn))
                        r_buf.copy_(r_out)

                def seg_b_body(layer=layer, h_attn=h_attn, r_buf=r_buf, write=write):
                    hn2, r2 = layer.post_attention_layernorm(h_attn, r_buf)
                    write.copy_(layer.mlp(hn2))
                    r_buf.copy_(r2)

                if self.world_size > 1:
                    dist.barrier()
                graph_a = Graph()
                with graph_ctx(graph_a, pool):
                    seg_a_body()
                if pool is None:
                    pool = graph_a.pool()
                self.seg_a[li][bs] = graph_a
                if self.world_size > 1:
                    dist.barrier()
                    self.acc.synchronize()
                    dist.barrier()
                graph_b = Graph()
                with graph_ctx(graph_b, pool):
                    seg_b_body()
                self.seg_b[li][bs] = graph_b
                if self.world_size > 1:
                    dist.barrier()
                    self.acc.synchronize()
                print(f"[graph-seg] rank{self.rank} bucket={bs} layer={li} captured",
                      flush=True)
            if _SEG_SLOW > 0:
                dbg(f"bucket={bs} slow-start sleep {_SEG_SLOW}s")
                time.sleep(_SEG_SLOW)
        self._set_segment_defer(False)
        self.graph_pool = pool
        if capture_backend != eager_backend:
            self.set_moe_backend(eager_backend, capture_backend)
        report = self.memory_report()
        print(
            f"[graph-seg] rank{self.rank} layers={len(layers)} buckets={self.graph_bs} "
            f"graphs={2 * len(layers) * len(self.graph_bs)} "
            f"decode_moe_backend={capture_backend} "
            f"allocated={report['allocated_gib']:.2f} GiB",
            flush=True,
        )

    @torch.inference_mode()
    def _run_model_segment(self, input_ids: torch.Tensor, positions: torch.Tensor):
        bs = input_ids.size(0)
        cap = next(x for x in self.graph_bs if x >= bs)
        if _SEG_PAD_RATIO > 0 and cap > bs * (1.0 + _SEG_PAD_RATIO):
            # I33: padding rows execute real attention/MoE inside the replayed
            # graph; past this ratio the wasted compute outweighs the launch
            # savings, so eager is the better deal.
            return self.model.compute_logits(self.model(input_ids, positions))
        bufs = self.seg_bufs[cap]
        context = get_context()
        gv = self.graph_vars
        gv["positions"][:bs] = positions
        # Same scratch-slot contract as the whole-graph path (I22): padding
        # rows write KV into the reserved scratch block, never slot 0.
        gv["slot_mapping"].fill_(self.scratch_slot)
        gv["slot_mapping"][:bs] = context.slot_mapping
        gv["context_lens"].fill_(1)
        gv["context_lens"][:bs] = context.context_lens
        gv["block_tables"].fill_(self.scratch_block_id)
        gv["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        bufs["ping0"][:bs].copy_(self.model.model.embed_tokens(input_ids))
        self._set_segment_defer(True)
        for li in range(len(self.seg_a)):
            write = bufs["ping1"] if li % 2 == 0 else bufs["ping0"]
            self.seg_a[li][cap].replay()
            dist.all_reduce(bufs["h_attn"][:bs])
            self.seg_b[li][cap].replay()
            dist.all_reduce(write[:bs])
        self._set_segment_defer(False)
        final = bufs["ping0"] if len(self.seg_a) % 2 == 0 else bufs["ping1"]
        hidden, _ = self.model.model.norm(final[:bs], bufs["r"][:bs])
        return self.model.compute_logits(hidden)

    @torch.inference_mode()
    def capture_cudagraph(self):
        if _SEGMENT_GRAPH[0] and self.world_size > 1:
            self.capture_segment_graphs()
            return
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        # Capture with a *real* decode context, not all zeros: zero context_lens
        # hand SDPA an all-False mask, and zero slot_mapping makes every token
        # write slot 0.  Both are degenerate and one of them left the MUSA
        # stream in an error state before capture even started ("operation not
        # permitted" at capture_begin's synchronize).  Distinct slots and a
        # one-block context exercise the normal path; replay overwrites them.
        slot_mapping = torch.arange(max_bs, dtype=torch.int32)
        context_lens = torch.full(
            (max_bs,),
            min(self.block_size, config.max_model_len),
            dtype=torch.int32,
        )
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None
        Graph = self.acc.MUSAGraph if IS_MUSA else self.acc.CUDAGraph
        graph_ctx = self.acc.graph
        # The decode graphs embed whatever MoE kernel runs during capture, so a
        # backend that is fast but not capturable (grouped_mm) is swapped for a
        # capturable one (vllm_fused) here only, and restored afterwards: eager
        # prefill keeps the fast kernel and pays nothing for the graph.
        eager_backend = config.moe_backend
        capture_backend = config.decode_moe_backend or eager_backend
        if capture_backend != eager_backend:
            self.set_moe_backend(capture_backend)
        self.captured_moe_backend = capture_backend
        memory_before = self.memory_report()
        # Ascending: capture the small buckets first, so a bucket that runs out of
        # memory leaves the small buckets usable instead of losing the whole
        # capture.  (Measured: the graph pool itself is cheap -- ``MUSAGraph()``
        # and a tiny capture cost <0.1 GiB; the ~15.6 GiB seen on the 30B work up
        # is spent *inside* the vendor MoE call, see docs/HANDOFF-P6-next.md T1.)
        for bs in self.graph_bs:
            # MCCL collectives must be recorded by every rank at the same point
            # in the capture.  Without these barriers rank 0 can start capturing
            # while rank 1 is still in its eager warmup, and the capturing
            # stream fails with "operation not permitted" (TP2 reproduced it;
            # TP1 captures fine).  The standalone MCCL probe brackets capture
            # with barriers for the same reason.
            if self.world_size > 1:
                dist.barrier()
            graph = Graph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            report = self.memory_report()
            print(
                f"[graph] rank{self.rank} bucket={bs} warmup "
                f"allocated={report['allocated_gib']:.2f} reserved={report['reserved_gib']:.2f} GiB",
                flush=True,
            )
            if self.world_size > 1:
                dist.barrier()
                # A barrier only guarantees both ranks *entered* it.  Flush the
                # device too, so neither rank records the first collective
                # while the other is still draining its barrier: that race made
                # TP2 capture fail intermittently with "operation not
                # permitted" (three attempts: fail, fail, pass with an added
                # print that slowed the host down).
                self.acc.synchronize()
            with graph_ctx(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.world_size > 1:
                dist.barrier()
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            self.acc.synchronize()
            report = self.memory_report()
            print(
                f"[graph] rank{self.rank} bucket={bs} "
                f"allocated={report['allocated_gib']:.2f} reserved={report['reserved_gib']:.2f} "
                f"peak={report['peak_gib']:.2f} GiB",
                flush=True,
            )
            reset_context()

        if capture_backend != eager_backend:
            self.set_moe_backend(eager_backend, capture_backend)
        memory_after = self.memory_report()
        allocated_before = memory_before.get("allocated_gib") or 0.0
        allocated_after = memory_after.get("allocated_gib") or 0.0
        reserved_before = memory_before.get("reserved_gib") or 0.0
        reserved_after = memory_after.get("reserved_gib") or 0.0
        # A graph pool usually shows up as reserved (not "allocated") memory and
        # then shrinks again once instantiate() releases it, so report both and
        # let the frozen gate use the larger delta.
        self.graph_memory_delta_gib = allocated_after - allocated_before
        self.graph_memory_reserved_delta_gib = reserved_after - reserved_before
        print(
            f"[graph] rank{self.rank} buckets={self.graph_bs} "
            f"decode_moe_backend={capture_backend} eager_moe_backend={eager_backend} "
            f"attention_backend={config.attention_backend} "
            f"allocated={allocated_before:.2f}->{allocated_after:.2f} GiB "
            f"reserved={reserved_before:.2f}->{reserved_after:.2f} GiB "
            f"peak={memory_after.get('peak_gib'):.2f} GiB",
            flush=True,
        )

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
