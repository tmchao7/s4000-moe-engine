"""Decode determinism regression for the tiny MoE checkpoint.

The MUSA SDPA decode path used to consume uninitialized KV-block tails.  The
masked values could contain NaNs, so identical TP=1 runs produced different
tokens.  This test recreates the engine for every run and intentionally does
not clear the KV cache.

    PYTHONPATH=/root/moe-server-s4000 python3 tests/p2_determinism.py
"""
import argparse
import sys

sys.path.insert(0, "/root/moe-server-s4000")

MODEL_DIR = "/tmp/tiny_moe"
PROMPTS = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12], [42, 42, 42]]


def run_once(max_tokens: int):
    from nanovllm import LLM, SamplingParams
    from nanovllm.layers.rotary_embedding import get_rope

    get_rope.cache_clear()
    llm = LLM(
        MODEL_DIR,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=256,
        gpu_memory_utilization=0.5,
    )
    try:
        outputs = llm.generate(
            PROMPTS,
            SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True),
            use_tqdm=False,
        )
        return [output["token_ids"] for output in outputs]
    finally:
        llm.exit()
        get_rope.cache_clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=4)
    args = parser.parse_args()

    outputs = []
    for run_id in range(args.runs):
        output = run_once(args.max_tokens)
        print(f"[determinism] run={run_id}: {output}", flush=True)
        outputs.append(output)
    assert all(output == outputs[0] for output in outputs), outputs
    print(f"[determinism] {args.runs}/{args.runs} runs match", flush=True)


if __name__ == "__main__":
    main()
