"""Ground-truth reference: hand-written pure-PyTorch forward of the tiny MoE
checkpoint on CPU, using the engine's own loaded weights. Compares greedy
generation against the engine at TP=1 and TP=2.

    PYTHONPATH=/root/moe-server-s4000 python3 tests/p2_ground_truth.py
"""
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from types import SimpleNamespace

sys.path.insert(0, "/root/moe-server-s4000")

MODEL_DIR = "/tmp/tiny_moe"
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29750")


def load_engine_weights_cpu():
    """Load the tiny checkpoint into our model classes on CPU (gloo ws=1)."""
    dist.init_process_group("gloo", rank=0, world_size=1)
    from transformers import Qwen3MoeConfig
    from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM
    from nanovllm.utils.loader import load_model
    from nanovllm.layers.rotary_embedding import get_rope
    get_rope.cache_clear()
    cfg = Qwen3MoeConfig.from_pretrained(MODEL_DIR)
    torch.set_default_dtype(torch.float32)
    runtime = SimpleNamespace(
        expert_parallel_size=1,
        moe_backend="loop",
        expert_placement="linear",
        expert_map_path=None,
        collect_expert_stats=False,
    )
    model = Qwen3MoeForCausalLM(cfg, runtime)
    load_model(model, MODEL_DIR)   # coverage assert runs here
    dist.destroy_process_group()
    get_rope.cache_clear()
    return model, cfg


@torch.no_grad()
def reference_generate(model, cfg, prompt_ids, max_new_tokens, rope_base=10000.0):
    """Naive full-recompute greedy generation (no KV cache, no TP)."""
    sd = {k: v for k, v in model.state_dict().items()}
    eps = cfg.rms_norm_eps
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    d = cfg.head_dim
    top_k = cfg.num_experts_per_tok

    def rms(x, w):
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        return (xf * torch.rsqrt(var + eps)).to(x.dtype) * w

    def rope(x, pos, base):
        # x: (T, h, d); matches nanovllm RotaryEmbedding exactly
        T = x.shape[0]
        inv = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
        t = pos.float()
        freqs = torch.einsum("i,j->ij", t, inv)
        cos, sin = freqs.cos(), freqs.sin()
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        xf = x.float()
        x1, x2 = xf.chunk(2, dim=-1)
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        return torch.cat((y1, y2), dim=-1).to(x.dtype)

    ids = list(prompt_ids)
    out = []
    for _ in range(max_new_tokens):
        T = len(ids)
        x = F.embedding(torch.tensor(ids), sd["model.embed_tokens.weight"])  # (T, H)
        residual = None
        for l in range(n_layers):
            p = f"model.layers.{l}."
            h = rms(x, sd[p + "input_layernorm.weight"])
            qkv = F.linear(h, sd[p + "self_attn.qkv_proj.weight"])
            q, k, v = qkv.split([n_heads * d, n_kv * d, n_kv * d], dim=-1)
            q = q.view(T, n_heads, d)
            k = k.view(T, n_kv, d)
            v = v.view(T, n_kv, d)
            q = rms(q, sd[p + "self_attn.q_norm.weight"])
            k = rms(k, sd[p + "self_attn.k_norm.weight"])
            pos = torch.arange(T, dtype=torch.float32)
            q = rope(q, pos, rope_base)
            k = rope(k, pos, rope_base)
            idx = torch.arange(n_kv).repeat_interleave(n_heads // n_kv)
            k_e = k.index_select(1, idx)
            v_e = v.index_select(1, idx)
            qh = q.transpose(0, 1)
            kh = k_e.transpose(0, 1)
            vh = v_e.transpose(0, 1)
            scores = torch.matmul(qh.float(), kh.float().transpose(-2, -1)) * (d ** -0.5)
            causal = torch.ones(T, T, dtype=torch.bool).tril()
            scores = scores.masked_fill(~causal, float("-inf"))
            o = (torch.softmax(scores, -1) @ vh.float()).transpose(0, 1).reshape(T, -1)
            attn_out = F.linear(o, sd[p + "self_attn.o_proj.weight"])
            x = x + attn_out
            h = rms(x, sd[p + "post_attention_layernorm.weight"])
            router = F.linear(h.float(), sd[p + "mlp.gate.weight"].float())
            rw = torch.softmax(router, -1)
            tw, tid = torch.topk(rw, top_k, -1)
            tw = tw / tw.sum(-1, keepdim=True)
            moe = torch.zeros_like(h)
            for t in range(T):
                for j in range(top_k):
                    e = int(tid[t, j])
                    w = tw[t, j].item()
                    gu_w = sd[f"{p}mlp.gate_up_proj"][e].transpose(0, 1)
                    dn_w = sd[f"{p}mlp.down_proj"][e].transpose(0, 1)
                    gu = F.linear(h[t:t + 1], gu_w)
                    g_, u_ = gu.chunk(2, -1)
                    y = F.linear(F.silu(g_) * u_, dn_w)
                    moe[t] += y[0] * w
            x = x + moe
        h = rms(x, sd["model.norm.weight"])
        logits = F.linear(h[-1:], sd["lm_head.weight"])
        nxt = int(logits.float().argmax(-1))
        out.append(nxt)
        ids.append(nxt)
    return out


def main():
    model, cfg = load_engine_weights_cpu()
    rope_base = cfg.rope_parameters["rope_theta"] if hasattr(cfg, "rope_parameters") \
        else getattr(cfg, "rope_theta", 10000.0)
    prompts = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12], [42, 42, 42]]
    refs = [reference_generate(model, cfg, p, 16, rope_base) for p in prompts]
    print("[truth] reference:", refs)

    from nanovllm import LLM, SamplingParams
    sp = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)

    results = {}
    for tp in (1, 2):
        from nanovllm.layers.rotary_embedding import get_rope
        get_rope.cache_clear()
        llm = LLM(MODEL_DIR, enforce_eager=True, tensor_parallel_size=tp,
                  max_model_len=256, gpu_memory_utilization=0.5)
        outs = llm.generate(prompts, sp, use_tqdm=False)
        got = [o["token_ids"] for o in outs]
        results[tp] = got
        match = got == refs
        print(f"[truth] TP{tp}        : {got}")
        print(f"[truth] TP{tp} == ref : {match}")
        try:
            llm.exit()
        except Exception:
            pass
        get_rope.cache_clear()
    print(f"[truth] TP1 == TP2  : {results[1] == results[2]}")


if __name__ == "__main__":
    main()
