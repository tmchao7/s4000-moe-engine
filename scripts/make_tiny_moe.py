"""Generate the tiny float32 Qwen3-MoE checkpoint used by the GPU unit tests.

The original /tmp/tiny_moe was created ad hoc on the previous server and never
checked in; p2/p6 (and several archived probes) hard-code it, so this script
recreates it deterministically.  It is a random-weight checkpoint: the tests
that consume it compare *backends against each other* (token equality), not
outputs against a reference model.

    PYTHONPATH=. python3 scripts/make_tiny_moe.py [--out /tmp/tiny_moe]

Tokenizer files are copied from a real Qwen3 tokenizer directory (default
/root/autodl-tmp/models/Qwen3-0.6B) so AutoTokenizer works without network.
"""
import argparse
import os
import shutil

import torch
from safetensors.torch import save_file


def build_config() -> dict:
    # head_dim=16 is deliberate: triton_paged cannot run on it, which forces
    # the tiny tests to pin attention_backend="sdpa" (keeps them about MoE).
    return {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "hidden_size": 64,
        "head_dim": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "intermediate_size": 32,
        "norm_topk_prob": True,
        "decoder_sparse_step": 1,
        "vocab_size": 151936,
        "max_position_embeddings": 4096,
        "rope_theta": 1000000,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "use_cache": True,
        "torch_dtype": "float32",
        "eos_token_id": 151645,
        "pad_token_id": 151643,
    }


def build_state(cfg: dict, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    std = 0.02

    def rand(*shape):
        return torch.randn(*shape, generator=g).mul_(std)

    state = {}
    state["model.embed_tokens.weight"] = rand(cfg["vocab_size"], cfg["hidden_size"])
    state["lm_head.weight"] = rand(cfg["vocab_size"], cfg["hidden_size"])
    state["model.norm.weight"] = torch.ones(cfg["hidden_size"])
    for layer in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{layer}"
        state[f"{p}.input_layernorm.weight"] = torch.ones(cfg["hidden_size"])
        state[f"{p}.post_attention_layernorm.weight"] = torch.ones(cfg["hidden_size"])
        state[f"{p}.self_attn.q_proj.weight"] = rand(cfg["hidden_size"], cfg["hidden_size"])
        kv = cfg["num_key_value_heads"] * cfg["head_dim"]
        state[f"{p}.self_attn.k_proj.weight"] = rand(kv, cfg["hidden_size"])
        state[f"{p}.self_attn.v_proj.weight"] = rand(kv, cfg["hidden_size"])
        state[f"{p}.self_attn.o_proj.weight"] = rand(cfg["hidden_size"], cfg["hidden_size"])
        state[f"{p}.self_attn.q_norm.weight"] = torch.ones(cfg["head_dim"])
        state[f"{p}.self_attn.k_norm.weight"] = torch.ones(cfg["head_dim"])
        state[f"{p}.mlp.gate.weight"] = rand(cfg["num_experts"], cfg["hidden_size"])
        for expert in range(cfg["num_experts"]):
            e = f"{p}.mlp.experts.{expert}"
            mid = cfg["moe_intermediate_size"]
            state[f"{e}.gate_proj.weight"] = rand(mid, cfg["hidden_size"])
            state[f"{e}.up_proj.weight"] = rand(mid, cfg["hidden_size"])
            state[f"{e}.down_proj.weight"] = rand(cfg["hidden_size"], mid)
    return {k: v.to(torch.float32) for k, v in state.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/tiny_moe")
    parser.add_argument("--tokenizer-from", default="/root/autodl-tmp/models/Qwen3-0.6B")
    args = parser.parse_args()

    import json

    os.makedirs(args.out, exist_ok=True)
    cfg = build_config()
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    save_file(build_state(cfg), os.path.join(args.out, "model.safetensors"))

    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        src = os.path.join(args.tokenizer_from, name)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(args.out, name))
    missing = [n for n in ("tokenizer.json", "tokenizer_config.json")
               if not os.path.isfile(os.path.join(args.out, n))]
    assert not missing, f"tokenizer files missing from {args.tokenizer_from}: {missing}"
    print(f"[make-tiny-moe] wrote {args.out} "
          f"({sum(v.numel() for v in build_state(cfg).values()) * 4 / 1e6:.1f} MB fp32)")


if __name__ == "__main__":
    main()
