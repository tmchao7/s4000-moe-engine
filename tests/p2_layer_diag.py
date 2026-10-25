"""P2 diagnosis: our Qwen3Moe layer forward vs HF Qwen3MoeDecoderLayer forward.

Loads layer-0 weights from the FP8 checkpoint with OUR dequant path, feeds the
SAME dequantized weights into HF's Qwen3MoeDecoderLayer, runs both forwards on
identical inputs and reports where they diverge.

    PYTHONPATH=/root/moe-server-s4000 python3 tests/p2_layer_diag.py
"""
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, "/root/moe-server-s4000")
MODEL = "/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8"


def main():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29611")
    dist.init_process_group("gloo", rank=0, world_size=1)
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer as HFLayer
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding
    from safetensors import safe_open
    from nanovllm.models.qwen3_moe import Qwen3MoeDecoderLayer as OurLayer
    from nanovllm.utils.loader import dequant_fp8_block

    cfg = Qwen3MoeConfig.from_pretrained(MODEL)
    hf = HFLayer(cfg, layer_idx=0).to(torch.bfloat16).eval()

    # collect layer-0 tensors from shard files
    tensors = {}
    import glob
    for f in glob.glob(os.path.join(MODEL, "*.safetensors")):
        with safe_open(f, "pt", "cpu") as sf:
            for k in sf.keys():
                if ".layers.0." in k:
                    tensors[k] = sf.get_tensor(k)

    def dq(name):
        return dequant_fp8_block(tensors[name], tensors[name + "_scale_inv"], [128, 128]).to(torch.bfloat16)

    sd = hf.state_dict()
    with torch.no_grad():
        # attention: q/k/v/o (fp8 -> dequant)
        for src, dst in [("self_attn.q_proj", "self_attn.q_proj"), ("self_attn.k_proj", "self_attn.k_proj"),
                         ("self_attn.v_proj", "self_attn.v_proj"), ("self_attn.o_proj", "self_attn.o_proj")]:
            w = dq(f"model.layers.0.{src}.weight")
            hf_sd_key = f"{dst}.weight"
            sd[hf_sd_key].copy_(w)
        for norm in ["input_layernorm", "post_attention_layernorm"]:
            sd[f"{norm}.weight"].copy_(tensors[f"model.layers.0.{norm}.weight"])
        sd["self_attn.q_norm.weight"].copy_(tensors["model.layers.0.self_attn.q_norm.weight"])
        sd["self_attn.k_norm.weight"].copy_(tensors["model.layers.0.self_attn.k_norm.weight"])
        # router
        sd["mlp.gate.weight"].copy_(tensors["model.layers.0.mlp.gate.weight"])
        # experts: HF stores merged (num_experts, ...) tensors
        n_exp = cfg.num_experts
        gu = torch.empty(n_exp, 2 * cfg.moe_intermediate_size, cfg.hidden_size, dtype=torch.bfloat16)
        dn = torch.empty(n_exp, cfg.hidden_size, cfg.moe_intermediate_size, dtype=torch.bfloat16)
        for e in range(n_exp):
            gu[e, :cfg.moe_intermediate_size] = dq(f"model.layers.0.mlp.experts.{e}.gate_proj.weight")
            gu[e, cfg.moe_intermediate_size:] = dq(f"model.layers.0.mlp.experts.{e}.up_proj.weight")
            dn[e] = dq(f"model.layers.0.mlp.experts.{e}.down_proj.weight")
        sd["mlp.experts.gate_up_proj"].copy_(gu)
        sd["mlp.experts.down_proj"].copy_(dn)
    hf.load_state_dict(sd)

    # our layer (CPU, bf16)
    torch.set_default_dtype(torch.bfloat16)
    ours = OurLayer(cfg).eval()
    from nanovllm.layers.rotary_embedding import get_rope
    get_rope.cache_clear()
    with torch.no_grad():
        ours.self_attn.qkv_proj.weight.data.copy_(torch.cat([
            dq("model.layers.0.self_attn.q_proj.weight"),
            dq("model.layers.0.self_attn.k_proj.weight"),
            dq("model.layers.0.self_attn.v_proj.weight")], dim=0))
        ours.self_attn.o_proj.weight.data.copy_(dq("model.layers.0.self_attn.o_proj.weight"))
        ours.self_attn.q_norm.weight.data.copy_(tensors["model.layers.0.self_attn.q_norm.weight"])
        ours.self_attn.k_norm.weight.data.copy_(tensors["model.layers.0.self_attn.k_norm.weight"])
        ours.input_layernorm.weight.data.copy_(tensors["model.layers.0.input_layernorm.weight"])
        ours.post_attention_layernorm.weight.data.copy_(tensors["model.layers.0.post_attention_layernorm.weight"])
        ours.mlp.gate.weight.data.copy_(tensors["model.layers.0.mlp.gate.weight"])
        for e in range(n_exp):
            g = dq(f"model.layers.0.mlp.experts.{e}.gate_proj.weight")
            u = dq(f"model.layers.0.mlp.experts.{e}.up_proj.weight")
            d = dq(f"model.layers.0.mlp.experts.{e}.down_proj.weight")
            ours.mlp.experts[e].gate_up_proj.data.copy_(torch.cat([g, u], dim=0))
            ours.mlp.experts[e].down_proj.data.copy_(d)

    # same input
    torch.manual_seed(7)
    T, H = 13, cfg.hidden_size
    x = torch.randn(T, H, dtype=torch.bfloat16)
    pos = torch.arange(T)
    causal_mask = torch.full((T, T), float("-inf"), dtype=torch.bfloat16).triu(1)

    rotary = Qwen3MoeRotaryEmbedding(cfg)
    with torch.no_grad():
        pos_emb = rotary(x.unsqueeze(0), pos.unsqueeze(0))
        h_hf = hf(x.unsqueeze(0),
                  position_ids=pos.unsqueeze(0),
                  attention_mask=causal_mask.unsqueeze(0).unsqueeze(0),
                  position_embeddings=pos_emb)
        h_hf = h_hf[0] if isinstance(h_hf, tuple) else h_hf
        h_hf = h_hf.squeeze(0)

        # our attention path: emulate engine (varlen loop + rope)
        from nanovllm.utils.context import set_context, reset_context
        cu = torch.tensor([0, T], dtype=torch.int32)
        set_context(True, cu, cu, T, T, torch.arange(T, dtype=torch.int32), None, None)
        h_ours_mlp, res_ours = ours(pos, x, None)
        h_ours = h_ours_mlp + res_ours    # HF layer returns residual + mlp output
        reset_context()

    diff = (h_hf.float() - h_ours.float()).abs()
    rel = diff.max() / h_hf.float().abs().max()
    print(f"[diag] layer0 output: max_abs={diff.max():.5f} mean_abs={diff.mean():.6f} rel={rel:.4f}")
    print(f"[diag] hf head:   {h_hf[0, :6].tolist()}")
    print(f"[diag] ours head: {h_ours[0, :6].tolist()}")

    # ---- submodule probes -------------------------------------------------
    with torch.no_grad():
        # MoE block alone
        m_hf = hf.mlp(x)
        m_hf = m_hf[0] if isinstance(m_hf, tuple) else m_hf
        m_our = ours.mlp(x)
        d = (m_hf.float() - m_our.float()).abs()
        print(f"[diag] MLP: hf_std={m_hf.float().std():.4f} ours_std={m_our.float().std():.4f} max_diff={d.max():.5f}")

        # attention alone (HF eager call)
        a_hf = hf.self_attn(hidden_states=x.unsqueeze(0),
                            position_embeddings=pos_emb,
                            attention_mask=causal_mask.unsqueeze(0).unsqueeze(0))
        a_hf = a_hf[0] if isinstance(a_hf, tuple) else a_hf
        a_hf = a_hf.squeeze(0)
        a_our = ours.self_attn(pos, x)
        d = (a_hf.float() - a_our.float()).abs()
        print(f"[diag] ATT: hf_std={a_hf.float().std():.4f} ours_std={a_our.float().std():.4f} max_diff={d.max():.5f}")

        # router internals
        lg = hf.mlp.gate(x)
        rw_hf = torch.softmax(lg.float(), -1)
        tw, tid = torch.topk(rw_hf, 8, -1)
        tw = tw / tw.sum(-1, keepdim=True)
        print(f"[diag] router top8 ids[0]: {tid[0].tolist()}")
        print(f"[diag] router top8 w[0]:   {[round(v,3) for v in tw[0].tolist()]}")

        # single expert probe: top expert of token 0
        e = int(tid[0, 0])
        xin = x[0:1]
        gu_hf = torch.nn.functional.linear(xin, hf.mlp.experts.gate_up_proj[e].to(torch.bfloat16))
        import torch.nn.functional as Fn
        g_, u_ = gu_hf.chunk(2, -1)
        y_hf = Fn.linear(Fn.silu(g_) * u_, hf.mlp.experts.down_proj[e].to(torch.bfloat16))
        y_our = ours.mlp.experts[e](xin)
        print(f"[diag] expert{e}: hf={y_hf[0,:4].tolist()} ours={y_our[0,:4].tolist()}")


if __name__ == "__main__":
    main()
