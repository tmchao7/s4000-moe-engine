"""End-to-end AsyncLLMEngine smoke test on the tiny MoE checkpoint.

Unlike ``p4_api.py``, this test starts the real TP workers and exercises the
background engine loop, concurrent admission, token delivery, cancellation,
and clean shutdown.
"""
import argparse
import asyncio
import sys

sys.path.insert(0, "/root/moe-server-s4000")

from nanovllm import SamplingParams
from nanovllm.async_engine import AsyncLLMEngine


async def drain(engine, handle):
    token_ids = []
    terminal = None
    while terminal is None:
        event = await asyncio.wait_for(engine.next_event(handle), timeout=30)
        if event.token_id is not None:
            token_ids.append(event.token_id)
        if event.finished:
            terminal = event
    return token_ids, terminal


async def run(args):
    engine = AsyncLLMEngine(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=2,
        expert_parallel_size=2,
        moe_backend="grouped_mm",
        scheduler_policy="stall_free",
        max_model_len=256,
        max_num_seqs=8,
        gpu_memory_utilization=0.2,
    )
    try:
        completed, cancelled = await asyncio.gather(
            engine.add_request(
                [1, 2, 3, 4],
                SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
            ),
            engine.add_request(
                [5, 6, 7, 8],
                SamplingParams(temperature=0, max_tokens=64, ignore_eos=True),
            ),
        )

        first = await asyncio.wait_for(engine.next_event(cancelled), timeout=30)
        assert first.token_id is not None and not first.finished, first
        engine.cancel(cancelled.seq_id)

        completed_tokens, completed_terminal = await drain(engine, completed)
        cancelled_tokens = [first.token_id]
        cancelled_terminal = None
        while cancelled_terminal is None:
            event = await asyncio.wait_for(engine.next_event(cancelled), timeout=30)
            if event.token_id is not None:
                cancelled_tokens.append(event.token_id)
            if event.finished:
                cancelled_terminal = event

        assert len(completed_tokens) == 8, completed_tokens
        assert completed_terminal.error is None
        assert not completed_terminal.cancelled
        assert cancelled_terminal.cancelled, cancelled_terminal
        metrics = engine.metrics()
        assert metrics["requests_total"] == 2, metrics
        assert metrics["requests_finished"] == 1, metrics
        assert metrics["requests_cancelled"] == 1, metrics
        assert metrics["active_requests"] == 0, metrics
        print(
            "[async] real TP2 concurrent completion/cancellation passed",
            {"completed": completed_tokens, "cancelled_prefix": cancelled_tokens, **metrics},
        )
    finally:
        await engine.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/tmp/tiny_moe")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
