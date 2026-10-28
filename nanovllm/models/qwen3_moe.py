import os

import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from transformers import Qwen3MoeConfig

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.models.qwen3 import Qwen3Attention
from nanovllm.utils.compat import is_capturing
from nanovllm.utils.context import get_context


# Runtime-switchable for interleaved A/B (docs/P8-moe-fusion-and-balance.md §3).
# Capacity shrink packs only owned (token, slot) pairs and sizes the dense bmm
# at counts.max() instead of the T*K worst case; it uses one host sync per
# layer and a data-dependent shape, so it is eager-only and defaults off.
_CAPACITY_SHRINK = [os.environ.get("NANOVLLM_MOE_CAPACITY_SHRINK", "0") == "1"]


def set_capacity_shrink(enabled: bool) -> None:
    _CAPACITY_SHRINK[0] = bool(enabled)


def capacity_shrink_enabled() -> bool:
    return _CAPACITY_SHRINK[0]


# Opt-in (max_load, n_owned) observations from the shrink path, for the P3.1
# S_max workload statistics.  Capped to keep long runs bounded.
_SHRINK_STATS = []
_SHRINK_STATS_ENABLED = [False]


def set_shrink_stats(enabled: bool) -> None:
    if not enabled:
        _SHRINK_STATS_ENABLED[0] = False
        return
    _SHRINK_STATS.clear()
    _SHRINK_STATS_ENABLED[0] = True


def shrink_stats():
    return list(_SHRINK_STATS)


class _ExpertView:
    """Compatibility view used by diagnostics and the reference loop backend."""

    def __init__(self, owner: "Qwen3MoeSparseMoeBlock", local_id: int):
        self.owner = owner
        self.local_id = local_id

    @property
    def gate_up_proj(self):
        return self.owner.gate_up_proj[self.local_id].transpose(0, 1)

    @property
    def down_proj(self):
        return self.owner.down_proj[self.local_id].transpose(0, 1)

    def __call__(self, x: torch.Tensor):
        gate_up = F.linear(x, self.gate_up_proj)
        return F.linear(self.owner.act_fn(gate_up), self.down_proj)


class _ExpertViews:

    def __init__(self, owner: "Qwen3MoeSparseMoeBlock"):
        self.owner = owner

    def __len__(self):
        return self.owner.experts_per_rank

    def __getitem__(self, local_id):
        if not 0 <= local_id < len(self):
            raise IndexError(local_id)
        return _ExpertView(self.owner, local_id)

    def __iter__(self):
        for local_id in range(len(self)):
            yield self[local_id]


class Qwen3MoeSparseMoeBlock(nn.Module):
    """Expert-parallel MoE: ranks own contiguous expert slices; router replicated.

    Every rank holds the full token hidden states (kept in sync by the TP
    allreduce after attention), so routing is computed locally and only the
    expert-output partial sums need one all_reduce.
    """

    def __init__(self, config: Qwen3MoeConfig, runtime_config, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.norm_topk_prob = config.norm_topk_prob
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.moe_backend = runtime_config.moe_backend
        # Decode may use a different backend than prefill: bmm_dense is faster
        # and capturable at the 1-64 token decode sizes, while its dense padding
        # is prohibitive for a thousands-token prefill chunk.  This is a
        # per-step choice (not graph-only), so it also applies when eager.
        self.decode_moe_backend = (
            runtime_config.decode_moe_backend or runtime_config.moe_backend
        )
        self.num_experts = config.num_experts
        self.collect_expert_stats = runtime_config.collect_expert_stats
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        ep_size = runtime_config.expert_parallel_size
        ep_rank = dist.get_rank()
        assert ep_size == dist.get_world_size()
        assert config.num_experts % ep_size == 0
        self.experts_per_rank = config.num_experts // ep_size
        self.expert_ids = self._expert_ids(
            config.num_experts,
            ep_size,
            ep_rank,
            runtime_config.expert_placement,
            runtime_config.expert_map_path,
            runtime_config.model,
        )
        self.expert_id_to_local = {expert_id: i for i, expert_id in enumerate(self.expert_ids)}
        global_to_local = torch.full((config.num_experts,), -1, dtype=torch.int64)
        for local_id, expert_id in enumerate(self.expert_ids):
            global_to_local[expert_id] = local_id
        self.register_buffer("global_to_local", global_to_local, persistent=False)
        self.register_buffer(
            "global_to_local_i32",
            global_to_local.to(torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "expert_counts",
            torch.zeros(config.num_experts, dtype=torch.int64),
            persistent=False,
        )
        # Fold-aware load stats (P3.1): counted BEFORE the backend folds
        # non-owned slots onto local expert 0, so expert 0's true load (its own
        # traffic plus every non-owned slot it silently absorbs) is visible.
        self.register_buffer(
            "owned_expert_counts",
            torch.zeros(self.experts_per_rank, dtype=torch.int64),
            persistent=False,
        )
        self.owned_slot_total = 0
        self.folded_slot_total = 0

        # Runtime orientation is [expert, input, output], exactly what
        # torch._grouped_mm consumes.  Keeping this as the only expert storage
        # avoids a second ~model-sized packed copy.
        self.gate_up_proj = nn.Parameter(torch.empty(
            self.experts_per_rank,
            config.hidden_size,
            2 * config.moe_intermediate_size,
        ))
        self.down_proj = nn.Parameter(torch.empty(
            self.experts_per_rank,
            config.moe_intermediate_size,
            config.hidden_size,
        ))
        self.act_fn = SiluAndMul()
        self.experts = _ExpertViews(self)
        self._loaded_expert_parts = set()
        self.profiler = None

    def _profile(self, name):
        if self.profiler is None:
            return nullcontext()
        return self.profiler.range(name, layer=self.layer_id)

    def _expert_ids(self, num_experts, ep_size, ep_rank, placement, map_path, model_path):
        if placement == "linear":
            start = ep_rank * self.experts_per_rank
            return list(range(start, start + self.experts_per_rank))
        if placement == "round_robin":
            return list(range(ep_rank, num_experts, ep_size))
        with open(map_path) as f:
            mapping = json.load(f)
        expected_hash = mapping.get("model_config_sha256")
        if expected_hash:
            actual_hash = hashlib.sha256(
                (Path(model_path) / "config.json").read_bytes()
            ).hexdigest()
            assert actual_hash == expected_hash, (
                "profiled expert map was generated for a different model config"
            )
        layers = mapping.get("layers", mapping)
        layer_mapping = layers.get(str(self.layer_id), layers.get(self.layer_id))
        assert layer_mapping is not None, f"missing profiled expert map for layer {self.layer_id}"
        expert_ids = layer_mapping.get(str(ep_rank), layer_mapping.get(ep_rank))
        assert len(expert_ids) == self.experts_per_rank
        assert len(set(expert_ids)) == len(expert_ids)
        return list(expert_ids)

    def owns_expert(self, global_id: int) -> bool:
        return global_id in self.expert_id_to_local

    @torch.no_grad()
    def load_expert_weight(self, global_id: int, part: str, loaded_weight: torch.Tensor):
        local_id = self.expert_id_to_local[global_id]
        if part == "gate_proj":
            target = self.gate_up_proj[local_id, :, :self.intermediate_size]
        elif part == "up_proj":
            target = self.gate_up_proj[local_id, :, self.intermediate_size:]
        elif part == "down_proj":
            target = self.down_proj[local_id]
        else:
            raise ValueError(f"unknown expert weight part: {part}")
        target.copy_(loaded_weight.transpose(0, 1))
        self._loaded_expert_parts.add((local_id, part))

    def validate_expert_weights_loaded(self):
        expected = {
            (local_id, part)
            for local_id in range(self.experts_per_rank)
            for part in ("gate_proj", "up_proj", "down_proj")
        }
        missing = expected - self._loaded_expert_parts
        assert not missing, (
            f"layer {self.layer_id}: {len(missing)} local expert weight parts "
            f"were not loaded, e.g. {sorted(missing)[:8]}"
        )

    def _forward_loop(self, hidden_states, topk_ids, topk_weights):
        num_tokens = hidden_states.size(0)
        out = torch.zeros_like(hidden_states)
        flat_ids = topk_ids.reshape(-1)
        flat_weights = topk_weights.reshape(-1)
        token_idx = torch.arange(num_tokens, device=hidden_states.device).unsqueeze(1).expand_as(topk_ids).reshape(-1)
        for local_id, expert in enumerate(self.experts):
            expert_id = self.expert_ids[local_id]
            mask = flat_ids == expert_id
            # ``mask.any()`` is a host sync; while a graph is being captured the
            # per-expert branch must go (empty index_select/index_add_ are no-ops).
            if not is_capturing() and not bool(mask.any()):
                continue
            idx = mask.nonzero(as_tuple=True)[0]
            tokens = token_idx[idx]
            x = hidden_states.index_select(0, tokens)
            out.index_add_(0, tokens, expert(x) * flat_weights[idx].unsqueeze(-1))
        return out

    def _forward_grouped_mm(self, hidden_states, topk_ids, topk_weights, fused=False):
        num_tokens = hidden_states.size(0)
        out = torch.zeros_like(hidden_states)
        with self._profile("moe.local_map"):
            local_raw = self.global_to_local[topk_ids].reshape(-1)
            # Non-local experts are folded onto expert 0 with weight 0 instead of
            # being filtered out.  ``local_raw[valid]`` (boolean indexing) and
            # ``bool(valid.any())`` are host syncs; 48 layers of those cost more
            # than the few extra rows expert 0 computes, and removing them is
            # also what makes this route/pack section capture-safe.
            owned = local_raw >= 0
            local_ids = local_raw.clamp(min=0)
        with self._profile("moe.route_pack"):
            device = hidden_states.device
            flat_token = torch.arange(num_tokens, device=device).unsqueeze(1).expand_as(topk_ids).reshape(-1)
            flat_slot = torch.arange(num_tokens * self.top_k, device=device)
            weights = topk_weights.reshape(-1) * owned.to(topk_weights.dtype)
            order = torch.argsort(local_ids)
            local_ids = local_ids[order]
            flat_token = flat_token[order]
            flat_slot = flat_slot[order]
            weights = weights[order]
            tokens = hidden_states.index_select(0, flat_token)
            counts = torch.zeros(
                self.experts_per_rank, dtype=torch.int32, device=device
            )
            counts.index_add_(
                0, local_ids, torch.ones_like(local_ids, dtype=torch.int32)
            )
            offsets = counts.cumsum(0).to(torch.int32)
        if self.profiler is not None:
            # Profiling is opt-in and already perturbs short segments; this is
            # the only host sync left in the route, and only on that path.
            self.profiler.observe(
                "moe.local_pairs", int(owned.sum()), layer=self.layer_id,
                tokens=num_tokens,
            )
        with self._profile("moe.grouped_mm_1"):
            gate_up = torch._grouped_mm(tokens, self.gate_up_proj, offs=offsets)
        with self._profile("moe.activation"):
            if fused:
                from nanovllm.kernels.moe_fusion import fused_silu_and_mul
                activated = fused_silu_and_mul(gate_up)
            else:
                activated = self.act_fn(gate_up)
        with self._profile("moe.grouped_mm_2"):
            expert_out = torch._grouped_mm(activated, self.down_proj, offs=offsets)
        with self._profile("moe.weighted_combine"):
            if fused:
                from nanovllm.kernels.moe_fusion import deterministic_weighted_combine
                return deterministic_weighted_combine(
                    expert_out, weights, flat_slot, num_tokens, self.top_k
                )
            contribution = expert_out * weights.unsqueeze(-1)
        # Deterministic top-k combine.  ``out.index_add_`` would accumulate the
        # top_k contributions of the same token with atomics, which makes the
        # reduction order (and therefore greedy decoding at near-ties) depend on
        # kernel scheduling.  Scatter back to the (token, slot) layout without
        # accumulation and reduce over the slot dimension in a fixed order.
            dense = contribution.new_zeros(
                num_tokens, self.top_k, hidden_states.size(-1)
            )
            dense.view(num_tokens * self.top_k, -1)[flat_slot] = contribution
            return dense.sum(dim=1)

    def _forward_bmm_dense(self, hidden_states, topk_ids, topk_weights):
        """Exact dense-expert bmm over the packed ``[E, in, out]`` weights.

        Every local expert is padded to ``S = tokens*top_k`` slots (the exact
        worst case: one expert can receive every slot), so no token is dropped
        and no host-side capacity check is needed.  The padding costs 64x the
        useful rows, but ``torch.bmm`` runs at GEMM efficiency while
        ``torch._grouped_mm`` is fixed-overhead bound at decode sizes: measured
        on the production shapes this is 1.3-1.95x faster than grouped_mm
        (scripts/p6_moe_microbench.py) and, unlike ``_grouped_mm``, it is
        capturable, so decode can run from a MUSA graph.

        Weights are consumed in their native layout -- no transposed copy.
        """
        if _CAPACITY_SHRINK[0] and not is_capturing():
            return self._forward_bmm_dense_shrunk(hidden_states, topk_ids, topk_weights)
        num_tokens = hidden_states.size(0)
        slots = num_tokens * self.top_k
        device = hidden_states.device
        with self._profile("moe.local_map"):
            local_raw = self.global_to_local[topk_ids].reshape(-1)
            owned = local_raw >= 0
            local_ids = local_raw.clamp(min=0)
        with self._profile("moe.route_pack"):
            flat_token = torch.arange(num_tokens, device=device).unsqueeze(1).expand_as(topk_ids).reshape(-1)
            flat_slot = torch.arange(slots, device=device)
            flat_weights = topk_weights.reshape(-1) * owned.to(topk_weights.dtype)
            order = torch.argsort(local_ids)
            sorted_ids = local_ids[order]
            sorted_token = flat_token[order]
            sorted_slot = flat_slot[order]
            sorted_weights = flat_weights[order]
            counts = torch.zeros(
                self.experts_per_rank, dtype=torch.int32, device=device
            )
            counts.index_add_(
                0, sorted_ids, torch.ones_like(sorted_ids, dtype=torch.int32)
            )
            starts = counts.cumsum(0).to(torch.int64) - counts.to(torch.int64)
            ranks = torch.arange(slots, device=device) - starts[sorted_ids]
        with self._profile("moe.bmm_1"):
            x_slots = hidden_states.index_select(0, sorted_token)
            x_pad = x_slots.new_zeros(self.experts_per_rank, slots, self.hidden_size)
            x_pad[sorted_ids, ranks] = x_slots
            gate_up = torch.bmm(x_pad, self.gate_up_proj)
        with self._profile("moe.activation"):
            activated = self.act_fn(gate_up)
        with self._profile("moe.bmm_2"):
            expert_out = torch.bmm(activated, self.down_proj)
        with self._profile("moe.weighted_combine"):
            contribution = expert_out[sorted_ids, ranks] * sorted_weights.unsqueeze(-1)
            dense = contribution.new_zeros(num_tokens, self.top_k, self.hidden_size)
            dense.view(slots, -1)[sorted_slot] = contribution
            return dense.sum(dim=1)

    def _forward_bmm_dense_shrunk(self, hidden_states, topk_ids, topk_weights):
        """bmm_dense with owned-only packing and S = counts.max().

        The default path pads every local expert to the T*K worst case, so at
        b32 each expert computes 256 rows of which ~4 are owned.  Here the
        non-owned slots (folded onto expert 0 with weight 0) are compacted away
        with a dump-slot scatter (no boolean indexing, no data-dependent
        kernel), and the dense bmm runs at S = max per-expert owned load.

        Cost: one stacked host sync per layer (S and n_owned travel together).
        Not capturable (data-dependent shape); graph mode keeps the padded
        path.  Numerics: same per-row K-dim reduction, rows only change their
        M-position, so results match the padded path up to bmm tiling order
        (verified by the L2 soft gate, not bitwise).
        """
        num_tokens = hidden_states.size(0)
        slots = num_tokens * self.top_k
        device = hidden_states.device
        E = self.experts_per_rank
        with self._profile("moe.local_map"):
            local_raw = self.global_to_local[topk_ids].reshape(-1)
            owned = local_raw >= 0
        with self._profile("moe.route_pack"):
            flat_token = torch.arange(num_tokens, device=device).unsqueeze(1).expand_as(topk_ids).reshape(-1)
            flat_slot = torch.arange(slots, device=device)
            flat_weights = topk_weights.reshape(-1) * owned.to(topk_weights.dtype)
            # Compaction without boolean indexing: owned rows keep their
            # compacted position, non-owned rows scatter into one dump slot
            # (duplicate indices land there; the slot is never read back).
            owned_i32 = owned.to(torch.int64)
            pos = torch.cumsum(owned_i32, 0) - 1
            pos = torch.where(owned, pos, torch.full_like(pos, slots))
            buf_local = torch.zeros(slots + 1, dtype=torch.int64, device=device)
            buf_token = torch.zeros(slots + 1, dtype=torch.int64, device=device)
            buf_slot = torch.zeros(slots + 1, dtype=torch.int64, device=device)
            buf_weight = torch.zeros(slots + 1, dtype=flat_weights.dtype, device=device)
            buf_local.scatter_(0, pos, local_raw.clamp(min=0))
            buf_token.scatter_(0, pos, flat_token)
            buf_slot.scatter_(0, pos, flat_slot)
            buf_weight.scatter_(0, pos, flat_weights)
            # Owned per-expert histogram; the dump bucket (id E) is dropped.
            counts = torch.zeros(E + 1, dtype=torch.int32, device=device)
            counts.scatter_add_(
                0, local_raw.clamp(min=0) + (~owned).to(torch.int64) * E,
                torch.ones_like(local_raw, dtype=torch.int32),
            )
            counts = counts[:E]
            # The single host sync of this path: S and n_owned together.
            host = torch.stack([counts.max(), counts.sum()]).to("cpu", non_blocking=False)
            max_load = int(host[0])
            n_owned = int(host[1])
            if _SHRINK_STATS_ENABLED[0] and len(_SHRINK_STATS) < 100000:
                _SHRINK_STATS.append((num_tokens, max_load, n_owned))
        if n_owned == 0:
            return torch.zeros_like(hidden_states)
        with self._profile("moe.bmm_1"):
            local_ids = buf_local[:n_owned]
            order = torch.argsort(local_ids, stable=True)
            sorted_ids = local_ids[order]
            sorted_token = buf_token[:n_owned][order]
            sorted_slot = buf_slot[:n_owned][order]
            sorted_weights = buf_weight[:n_owned][order]
            starts = counts.cumsum(0).to(torch.int64) - counts.to(torch.int64)
            ranks = torch.arange(n_owned, device=device) - starts[sorted_ids]
            x_slots = hidden_states.index_select(0, sorted_token)
            x_pad = x_slots.new_zeros(E, max_load, self.hidden_size)
            x_pad[sorted_ids, ranks] = x_slots
            gate_up = torch.bmm(x_pad, self.gate_up_proj)
        with self._profile("moe.activation"):
            activated = self.act_fn(gate_up)
        with self._profile("moe.bmm_2"):
            expert_out = torch.bmm(activated, self.down_proj)
        with self._profile("moe.weighted_combine"):
            contribution = expert_out[sorted_ids, ranks] * sorted_weights.unsqueeze(-1)
            dense = contribution.new_zeros(num_tokens, self.top_k, self.hidden_size)
            dense.view(slots, -1)[sorted_slot] = contribution
            return dense.sum(dim=1)

    def _forward_vllm_fused(self, hidden_states, topk_ids, topk_weights):
        """Reuse the vllm-musa Triton fused MoE instead of re-implementing it.

        The vendor kernel wants ``[E, out, in]`` weights and asserts
        ``stride(-1) == 1``, but our packed storage is ``[E, in, out]`` (what
        ``torch._grouped_mm`` consumes).  Transposing into a contiguous copy
        costs 576 MiB per layer -- 27 GiB for the 30B, which is what OOM'd every
        P6 capture attempt.  ``vendor_moe.fused_experts_zero_copy`` passes the
        transposed *view* instead and removes only those two asserts in memory;
        the Triton kernel reads strides explicitly and is numerically identical
        (tests/p6_vendor_zero_copy.py, including graph replay).

        The MUSA vendor fast path (S4000_ops: gemv/mutlass) aborts with an
        illegal memory access at our prefill sizes (>=1024 tokens), so the
        generic Triton kernel is forced.
        """
        if hidden_states.dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError(
                f"vllm_fused requires bf16/fp16 activations, got {hidden_states.dtype}"
            )
        from nanovllm.kernels.vendor_moe import fused_experts_zero_copy

        with self._profile("moe.vllm_fused"):
            return fused_experts_zero_copy(
                hidden_states,
                self.gate_up_proj,
                self.down_proj,
                topk_weights,
                topk_ids,
                global_num_experts=self.num_experts,
                expert_map=self.global_to_local_i32,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with self._profile("moe.router_topk"):
            router_logits = self.gate(hidden_states)
            routing_weights = torch.softmax(router_logits.float(), dim=-1)
            topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.collect_expert_stats:
            self.expert_counts.add_(
                torch.bincount(topk_ids.reshape(-1), minlength=self.expert_counts.numel())
            )
            if not is_capturing():
                local_raw = self.global_to_local[topk_ids].reshape(-1)
                owned = local_raw >= 0
                # fold-前 per-local-expert owned load; expert_counts above is
                # the global view (same on both ranks), this is the rank view.
                self.owned_expert_counts.add_(
                    torch.bincount(
                        local_raw.clamp(min=0)[owned],
                        minlength=self.experts_per_rank,
                    )
                )
                self.owned_slot_total += int(owned.sum())
                self.folded_slot_total += int((~owned).sum())
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(hidden_states.dtype)

        backend = self.moe_backend if get_context().is_prefill else self.decode_moe_backend
        if backend == "vllm_fused":
            out = self._forward_vllm_fused(hidden_states, topk_ids, topk_weights)
        elif backend == "bmm_dense":
            out = self._forward_bmm_dense(hidden_states, topk_ids, topk_weights)
        elif backend in ("grouped_mm", "fused_grouped_mm"):
            out = self._forward_grouped_mm(
                hidden_states, topk_ids, topk_weights,
                fused=backend == "fused_grouped_mm",
            )
        else:
            out = self._forward_loop(hidden_states, topk_ids, topk_weights)
        with self._profile("moe.ep_all_reduce"):
            # defer_reduce: P10 segment-graph prototype — this combine moves
            # out of the captured segment; run_model all_reduces eagerly.
            if not getattr(self, "defer_reduce", False):
                dist.all_reduce(out)
        return out


class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(self, config: Qwen3MoeConfig, runtime_config, layer_id: int):
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", True),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MoeSparseMoeBlock(config, runtime_config, layer_id)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3MoeModel(nn.Module):

    def __init__(self, config: Qwen3MoeConfig, runtime_config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3MoeDecoderLayer(config, runtime_config, layer_id)
            for layer_id in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3MoeConfig, runtime_config):
        super().__init__()
        self.model = Qwen3MoeModel(config, runtime_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)

    def reset_expert_stats(self):
        for layer in self.model.layers:
            layer.mlp.expert_counts.zero_()
            layer.mlp.owned_expert_counts.zero_()
            layer.mlp.owned_slot_total = 0
            layer.mlp.folded_slot_total = 0

    def expert_stats(self):
        return torch.stack([layer.mlp.expert_counts for layer in self.model.layers])

    def owned_expert_stats(self):
        return {
            "owned_counts": torch.stack(
                [layer.mlp.owned_expert_counts for layer in self.model.layers]
            ),
            "owned_slot_total": [layer.mlp.owned_slot_total for layer in self.model.layers],
            "folded_slot_total": [layer.mlp.folded_slot_total for layer in self.model.layers],
        }
