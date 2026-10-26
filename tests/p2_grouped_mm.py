"""Validate the packed grouped-MM MoE backend against the reference loop.

    PYTHONPATH=/root/moe-server-s4000 python3 tests/p2_grouped_mm.py
"""
import argparse
import sys

sys.path.insert(0, "/root/moe-server-s4000")

MODEL_DIR = "/tmp/tiny_moe"
PROMPTS = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12], [42, 42, 42]]


def run(tp: int, backend: str, max_tokens: int, placement: str = "linear"):
    from nanovllm import LLM, SamplingParams
    from nanovllm.layers.rotary_embedding import get_rope

    get_rope.cache_clear()
    llm = LLM(
        MODEL_DIR,
        enforce_eager=True,
        tensor_parallel_size=tp,
        expert_parallel_size=tp,
        moe_backend=backend,
        expert_placement=placement,
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
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--determinism-repeats", type=int, default=2)
    args = parser.parse_args()

    results = {}
    for tp in (1, 2):
        for backend in ("loop", "grouped_mm", "fused_grouped_mm"):
            key = f"tp{tp}-{backend}"
            results[key] = run(tp, backend, args.max_tokens)
            print(f"[grouped-mm] {key}: {results[key]}", flush=True)
    results["tp2-grouped_mm-round_robin"] = run(
        2, "grouped_mm", args.max_tokens, placement="round_robin"
    )
    print(
        "[grouped-mm] tp2-grouped_mm-round_robin:",
        results["tp2-grouped_mm-round_robin"],
        flush=True,
    )

    reference = results["tp1-loop"]
    for key, output in results.items():
        assert output == reference, f"{key} diverged: {output} != {reference}"
    print("[grouped-mm] loop/grouped_mm/fused and TP1/TP2 all match", flush=True)

    # Guard the deterministic top-k combine: the same engine config must give
    # the same tokens when rebuilt (a non-deterministic reduction would show up
    # as an occasional flip at a near-tie).
    repeats = [
        run(2, "grouped_mm", args.max_tokens)
        for _ in range(args.determinism_repeats)
    ]
    assert all(item == repeats[0] for item in repeats), repeats
    print(
        f"[grouped-mm] grouped_mm repeat determinism ok ({args.determinism_repeats}x)",
        flush=True,
    )


if __name__ == "__main__":
    main()
