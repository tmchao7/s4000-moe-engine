"""Layer-by-layer diff: engine TP1 prefill vs hand-written CPU reference.

Same process: loads the engine, runs prefill on GPU with hooks on every
DecoderLayer output; then computes the hand-written reference forward on CPU
with the same weights; diffs layer-by-layer hidden states.

    PYTHONPATH=/root/moe-server-s4000 python3 tests/p2_layer_by_layer.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/moe-server-s4000")

MODEL_DIR = "/tmp/tiny_moe"


def rms_ref(x, w, eps):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps)).to(x.dtype) * w


def rope_ref(x, pos, base, d):
    T = x.shape[0]
    inv = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    freqs = torch.einsum("i,j->ij", pos.float(), inv)
    cos, sin = freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)
    xf = x.float()
    x1, x2 = xf.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).to(x.dtype)


@torch.no_grad()
def main():
    # ---- load engine -------------------------------------------------------
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.sequence import Sequence
    llm = LLM(MODEL_DIR, enforce_eager=True, tensor_parallel_size=1,
              max_model_len=256, gpu_memory_utilization=0.5)
    mr = llm.model_runner
    model = mr.model
    cfg = mr.config.hf_config
    eps = cfg.rms_norm_eps
    base = cfg.rope_parameters["rope_theta"] if hasattr(cfg, "rope_parameters") \
        else 10000.0
    n_heads, n_kv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    top_k = cfg.num_experts_per_tok

    prompt = [1, 2, 3, 4, 5, 6, 7, 8]
    T = len(prompt)
    seq = Sequence(prompt, SamplingParams(temperature=0, max_tokens=4, ignore_eos=True))
    seq.num_scheduled_tokens = T
    llm.scheduler.add(seq)
    seqs, mode = llm.scheduler.schedule()

    # capture input embeddings and every layer output
    caps = []
    emb = model.model.embed_tokens
    orig_emb = emb.forward
    def hooked_emb(x):
        caps.append(("embed", x.detach().clone()))
        return orig_emb(x)
    emb.forward = hooked_emb
    layer_hooks = []
    for l, layer in enumerate(model.model.layers):
        def mk(idx):
            def h(mod, inp, out):
                caps.append((f"layer{idx}", out[0].detach().clone()))
            return h
        layer_hooks.append(layer.register_forward_hook(mk(l)))
    mr.run(seqs, mode)
    for hk in layer_hooks:
        hk.remove()
    emb.forward = orig_emb

    # ---- hand-written reference on CPU using engine weights ----------------
    # pull weights to CPU (contiguous, synced)
    def w(name):
        t = dict(model.named_parameters())[name]
        torch.musa.synchronize()
        return t.detach().float().cpu().contiguous().clone()

    sd = {}
    for name, _ in model.named_parameters():
        sd[name] = w(name)

    x = caps[0][1]
    if x.dim() == 1:      # VocabParallelEmbedding hook captured input ids
        x = caps[1][1]
        embed_cap = 1
    else:
        embed_cap = 0
    x = x.float().cpu().clone()   # engine embeddings as input
    print(f"[lay] embed x[0,:4]: {[round(v,4) for v in x[0,:4].tolist()]}")
    print(f"[lay] embed x[7,:4]: {[round(v,4) for v in x[7,:4].tolist()]}")

    for l in range(cfg.num_hidden_layers):
        p = f"model.layers.{l}."
        h = rms_ref(x, sd[p + "input_layernorm.weight"], eps)
        qkv = F.linear(h, sd[p + "self_attn.qkv_proj.weight"])
        q, k, v = qkv.split([n_heads * d, n_kv * d, n_kv * d], dim=-1)
        q = q.view(T, n_heads, d)
        k = k.view(T, n_kv, d)
        v = v.view(T, n_kv, d)
        q = rms_ref(q, sd[p + "self_attn.q_norm.weight"], eps)
        k = rms_ref(k, sd[p + "self_attn.k_norm.weight"], eps)
        pos = torch.arange(T, dtype=torch.float32)
        q = rope_ref(q, pos, base, d)
        k = rope_ref(k, pos, base, d)
        idx = torch.arange(n_kv).repeat_interleave(n_heads // n_kv)
        k_e = k.index_select(1, idx)
        v_e = v.index_select(1, idx)
        qh, kh, vh = q.transpose(0, 1), k_e.transpose(0, 1), v_e.transpose(0, 1)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) * (d ** -0.5)
        causal = torch.ones(T, T, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        o = (torch.softmax(scores, -1) @ vh).transpose(0, 1).reshape(T, -1)
        attn_out = F.linear(o, sd[p + "self_attn.o_proj.weight"])
        x = x + attn_out
        h = rms_ref(x, sd[p + "post_attention_layernorm.weight"], eps)
        router = F.linear(h.float(), sd[p + "mlp.gate.weight"].float())
        rw = torch.softmax(router, -1)
        tw, tid = torch.topk(rw, top_k, -1)
        tw = tw / tw.sum(-1, keepdim=True)
        moe = torch.zeros_like(h)
        for t in range(T):
            for j in range(top_k):
                e = int(tid[t, j])
                gu = F.linear(
                    h[t:t + 1],
                    sd[f"{p}mlp.gate_up_proj"][e].transpose(0, 1),
                )
                g_, u_ = gu.chunk(2, -1)
                y = F.linear(
                    F.silu(g_) * u_,
                    sd[f"{p}mlp.down_proj"][e].transpose(0, 1),
                )
                moe[t] += y[0] * tw[t, j].item()
        x = x + moe

        eng_out = caps[l + 1 + embed_cap][1].float().cpu()
        dd = (x - eng_out).abs()
        pr = dd.max(dim=1).values
        if l < 3 or l == cfg.num_hidden_layers - 1:
            print(f"[lay] layer{l}: max={dd.max().item():.5f} "
                  f"row0={pr[0].item():.5f} row1={pr[1].item():.5f} row7={pr[7].item():.5f} "
                  f"eng_row1={[round(v,3) for v in eng_out[1,:3].tolist()]} ref_row1={[round(v,3) for v in x[1,:3].tolist()]}")
        if l == 0 and dd.max().item() > 1e-3:
            # drill into row1: recompute ref attention pieces for row1 and
            # compare against engine row1 hidden step by step
            print(f"[lay] L0 ref x1(attn) vs eng:")
            break

    try:
        llm.exit()
    except Exception:
        pass


if __name__ == "__main__":
    main()
