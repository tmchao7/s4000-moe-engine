"""P2 check 1: expert-parallel TP correctness on a tiny random MoE model.

Generates a tiny Qwen3-MoE checkpoint with random weights (single process,
gloo), then compares greedy outputs of our engine at TP=1 vs TP=2.
They must be token-identical.

    PYTHONPATH=/root/moe-server-s4000 python3 tests/p2_tp_check.py
"""
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, "/root/moe-server-s4000")

MODEL_DIR = "/tmp/tiny_moe"


def build_tiny_checkpoint():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29600")
    dist.init_process_group("gloo", rank=0, world_size=1)
    from transformers import Qwen3MoeConfig
    from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM

    config = Qwen3MoeConfig(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=32, vocab_size=128, max_position_embeddings=512,
        rms_norm_eps=1e-6, rope_theta=10000.0, attention_bias=False,
        tie_word_embeddings=False, norm_topk_prob=True, hidden_act="silu",
        architectures=["Qwen3MoeForCausalLM"], torch_dtype=torch.float32,
    )
    torch.manual_seed(42)
    model = Qwen3MoeForCausalLM(config)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.5)
    # save HF-style names (gate_up_proj split back into gate_proj/up_proj)
    state = {}
    inter = config.moe_intermediate_size
    for k, v in model.state_dict().items():
        if "gate_up_proj" in k:
            base = k.replace("gate_up_proj.weight", "")
            state[base + "gate_proj.weight"] = v[:inter].contiguous()
            state[base + "up_proj.weight"] = v[inter:].contiguous()
        else:
            state[k] = v.contiguous()
    from safetensors.torch import save_file
    save_file(state, os.path.join(MODEL_DIR, "model.safetensors"))
    config.save_pretrained(MODEL_DIR)
    print(f"[tp-check] tiny checkpoint saved to {MODEL_DIR} ({len(state)} tensors)")
    dist.destroy_process_group()
    # build ran on CPU and polluted get_rope's lru_cache with a CPU rope module
    from nanovllm.layers.rotary_embedding import get_rope
    get_rope.cache_clear()


def run_llm(tp):
    from nanovllm import LLM, SamplingParams
    prompts = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12], [42, 42, 42]]
    llm = LLM(MODEL_DIR, enforce_eager=True, tensor_parallel_size=tp,
              max_model_len=256, gpu_memory_utilization=0.5)
    sp = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    try:
        llm.exit()
    except Exception:
        pass
    return [o["token_ids"] for o in outs]


def main():
    if not os.path.isfile(os.path.join(MODEL_DIR, "config.json")):
        build_tiny_checkpoint()
    out1 = run_llm(1)
    print("[tp-check] TP1 outputs:", out1, flush=True)
    out2 = run_llm(2)
    print("[tp-check] TP2 outputs:", out2, flush=True)
    match = out1 == out2
    print(f"[tp-check] TP1 == TP2: {match}")
    with open("/root/p2_tp_check.json", "w") as f:
        import json
        json.dump({"match": match, "tp1": out1, "tp2": out2}, f, indent=1)


if __name__ == "__main__":
    main()
