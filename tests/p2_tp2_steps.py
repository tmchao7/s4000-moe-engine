"""Per-step TP1 vs TP2 comparison on the tiny MoE model.

    PYTHONPATH=/root/moe-server-s4000 python3 tests/p2_tp2_steps.py
"""
import sys
sys.path.insert(0, "/root/moe-server-s4000")
from nanovllm import LLM, SamplingParams

MODEL_DIR = "/tmp/tiny_moe"
PROMPTS = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12], [42, 42, 42]]

def run(tp):
    from nanovllm.layers.rotary_embedding import get_rope
    get_rope.cache_clear()
    llm = LLM(MODEL_DIR, enforce_eager=True, tensor_parallel_size=tp,
              max_model_len=256, gpu_memory_utilization=0.5)
    sp = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
    outs = llm.generate(PROMPTS, sp, use_tqdm=False)
    try:
        llm.exit()
    except Exception:
        pass
    get_rope.cache_clear()
    return [o["token_ids"] for o in outs]

def main():
    t1 = run(1)
    t2 = run(2)
    for i, (a, b) in enumerate(zip(t1, t2)):
        for step, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f"seq{i}: first divergence at step {step}: TP1={x} TP2={y}")
                print(f"  TP1: {a}")
                print(f"  TP2: {b}")
                break
        else:
            print(f"seq{i}: all {len(a)} steps match")


if __name__ == "__main__":
    main()
