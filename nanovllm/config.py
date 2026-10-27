import os
from dataclasses import dataclass
from typing import Literal
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    distributed_init_method: str = ""
    shm_name: str = ""
    expert_parallel_size: int = 0
    moe_backend: Literal[
        "loop", "grouped_mm", "fused_grouped_mm", "bmm_dense", "vllm_fused"
    ] = "grouped_mm"
    # MoE backend for *decode* steps, independent of ``moe_backend`` (prefill).
    # ``None`` = use ``moe_backend`` everywhere.  Decode-only is needed when the
    # best decode kernel is capturable but poor at prefill sizes (bmm_dense),
    # or when the decode graph must embed a different kernel than eager prefill.
    decode_moe_backend: Literal[
        "loop", "grouped_mm", "fused_grouped_mm", "bmm_dense", "vllm_fused"
    ] | None = None
    attention_backend: Literal["sdpa", "triton_paged"] = "triton_paged"
    expert_placement: Literal["linear", "round_robin", "profiled"] = "linear"
    expert_map_path: str | None = None
    scheduler_policy: Literal["fcfs", "stall_free", "stall_free_lpm"] = "fcfs"
    prefill_chunk_size: int = 256
    scheduler_starvation_steps: int = 32
    collect_expert_stats: bool = False
    metrics_path: str | None = None
    enable_profiling: bool = False

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if self.expert_parallel_size == 0:
            self.expert_parallel_size = self.tensor_parallel_size
        assert self.expert_parallel_size == self.tensor_parallel_size, (
            "single-node hybrid TP+EP currently requires expert_parallel_size "
            "to equal tensor_parallel_size"
        )
        assert self.moe_backend in (
            "loop", "grouped_mm", "fused_grouped_mm", "bmm_dense", "vllm_fused"
        )
        assert self.decode_moe_backend in (
            None, "loop", "grouped_mm", "fused_grouped_mm", "bmm_dense", "vllm_fused",
        )
        assert self.attention_backend in ("sdpa", "triton_paged")
        assert self.expert_placement in ("linear", "round_robin", "profiled")
        assert self.scheduler_policy in ("fcfs", "stall_free", "stall_free_lpm")
        assert self.prefill_chunk_size > 0
        assert self.scheduler_starvation_steps > 0
        if self.expert_placement == "profiled":
            assert self.expert_map_path and os.path.isfile(self.expert_map_path)
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
