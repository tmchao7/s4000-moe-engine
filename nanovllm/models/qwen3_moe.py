import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import hashlib
import json
from pathlib import Path
from transformers import Qwen3MoeConfig

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.models.qwen3 import Qwen3Attention


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
            "expert_counts",
            torch.zeros(config.num_experts, dtype=torch.int64),
            persistent=False,
        )

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
            if not bool(mask.any()):
                continue
            idx = mask.nonzero(as_tuple=True)[0]
            tokens = token_idx[idx]
            x = hidden_states.index_select(0, tokens)
            out.index_add_(0, tokens, expert(x) * flat_weights[idx].unsqueeze(-1))
        return out

    def _forward_grouped_mm(self, hidden_states, topk_ids, topk_weights):
        num_tokens = hidden_states.size(0)
        out = torch.zeros_like(hidden_states)
        local_ids = self.global_to_local[topk_ids].reshape(-1)
        valid = local_ids >= 0
        if not bool(valid.any()):
            return out
        flat_token = torch.arange(num_tokens, device=hidden_states.device).unsqueeze(1).expand_as(topk_ids).reshape(-1)
        flat_slot = torch.arange(num_tokens * self.top_k, device=hidden_states.device)
        weights = topk_weights.reshape(-1)
        local_ids = local_ids[valid]
        flat_token = flat_token[valid]
        flat_slot = flat_slot[valid]
        weights = weights[valid]
        order = torch.argsort(local_ids)
        local_ids = local_ids[order]
        flat_token = flat_token[order]
        flat_slot = flat_slot[order]
        weights = weights[order]
        tokens = hidden_states.index_select(0, flat_token)
        counts = torch.bincount(local_ids, minlength=self.experts_per_rank)
        offsets = counts.cumsum(0).to(torch.int32)
        gate_up = torch._grouped_mm(tokens, self.gate_up_proj, offs=offsets)
        expert_out = torch._grouped_mm(self.act_fn(gate_up), self.down_proj, offs=offsets)
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits = self.gate(hidden_states)
        routing_weights = torch.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.collect_expert_stats:
            self.expert_counts.add_(
                torch.bincount(topk_ids.reshape(-1), minlength=self.expert_counts.numel())
            )
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(hidden_states.dtype)

        if self.moe_backend == "grouped_mm":
            out = self._forward_grouped_mm(hidden_states, topk_ids, topk_weights)
        else:
            out = self._forward_loop(hidden_states, topk_ids, topk_weights)
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

    def expert_stats(self):
        return torch.stack([layer.mlp.expert_counts for layer in self.model.layers])
