import pickle
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

        dist.init_process_group(
            DIST_BACKEND,
            config.distributed_init_method,
            world_size=self.world_size,
            rank=rank,
        )
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
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
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
            else:
                dist.barrier()
                self.shm = SharedMemory(name=config.shm_name)
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        self.acc.synchronize()
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
        for module in self.model.modules():
            if hasattr(module, "attention_backend"):
                module.attention_backend = backend

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
        print(
            f"[kvcache] rank{self.rank} blocks={config.num_kvcache_blocks} "
            f"block_bytes={block_bytes} kv={config.num_kvcache_blocks * block_bytes / 2**30:.2f} GiB "
            f"explicit={explicit_blocks} total={total / 2**30:.2f} GiB used={used / 2**30:.2f} GiB "
            f"peak={peak / 2**30:.2f} GiB current={current / 2**30:.2f} GiB free={free / 2**30:.2f} GiB",
            flush=True,
        )
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
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
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
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
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).to(self.device, non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, mode: str):
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
            # graph.  Fill their slots with 0, not -1: the captured index_put
            # writes every row of the bucket, and a -1 slot would index the
            # *last* KV element.  context_lens=1 keeps those rows on a valid
            # one-token attention instead of a 0/0 division.
            graph_vars["slot_mapping"].fill_(0)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].fill_(1)
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], mode: str) -> list[int]:
        with self.profiler.range("runner.prepare", mode=mode):
            input_ids, positions = self.prepare_decode(seqs) if mode == "decode" else self.prepare_prefill(seqs)
            temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        with self.profiler.range("runner.forward", mode=mode):
            logits = self.run_model(input_ids, positions, mode)
        with self.profiler.range("runner.sample", mode=mode):
            token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
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
