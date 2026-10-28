"""Minimal graph-capture repro on the tiny MoE (no 30B load).

Run with ``MUSA_LAUNCH_BLOCKING=1`` to turn the asynchronous
"operation not permitted" into a synchronous traceback naming the operator.

Run:
  MUSA_LAUNCH_BLOCKING=1 PYTHONPATH=. python3 tests/p6_graph_min.py --tp 1
"""
import argparse
import sys

sys.path.insert(0, "/root/moe-server-s4000")

MODEL_DIR = "/tmp/tiny_moe"
PROMPTS = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12], [42, 42, 42]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--model", default=MODEL_DIR)
    parser.add_argument("--dense", action="store_true", help="dense model: no MoE args")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--decode-moe-backend", default="bmm_dense")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--generates", type=int, default=1,
                        help="run the same generate N times (2+ tests replay reuse)")
    parser.add_argument("--clear-prefix-cache", action="store_true")
    parser.add_argument("--trace-allreduce", action="store_true")
    args = parser.parse_args()

    if args.trace_allreduce:
        import torch.distributed as dist

        original = dist.all_reduce
        counter = {"n": 0}

        def traced(tensor, *call_args, **call_kwargs):
            counter["n"] += 1
            print(
                f"[allreduce] rank={dist.get_rank()} n={counter['n']} "
                f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                f"contiguous={tensor.is_contiguous()}",
                flush=True,
            )
            return original(tensor, *call_args, **call_kwargs)

        dist.all_reduce = traced

    from nanovllm import LLM, SamplingParams

    print(f"[graph-min] model={args.model} tp={args.tp} attention={args.attention_backend} "
          f"dense={args.dense} decode_moe={args.decode_moe_backend}", flush=True)
    kwargs = {}
    if not args.dense:
        kwargs = {
            "moe_backend": "grouped_mm",
            "decode_moe_backend": args.decode_moe_backend,
        }
    llm = LLM(
        args.model,
        enforce_eager=False,
        tensor_parallel_size=args.tp,
        expert_parallel_size=args.tp,
        attention_backend=args.attention_backend,
        max_model_len=256 if args.dense is False else 2048,
        max_num_seqs=8,
        gpu_memory_utilization=0.3,
        **kwargs,
    )
    print(f"[graph-min] graph_report={llm.model_runner.call('graph_report')}", flush=True)
    for index in range(args.generates):
        if args.clear_prefix_cache:
            llm.scheduler.block_manager.clear_prefix_cache()
        print(f"[graph-min] generate {index + 1}/{args.generates}", flush=True)
        outputs = llm.generate(
            PROMPTS,
            SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True),
            use_tqdm=False,
        )
        print(f"[graph-min] tokens[{index}]={[o['token_ids'] for o in outputs]}", flush=True)
    llm.exit()
    print("[graph-min] OK", flush=True)


if __name__ == "__main__":
    main()
