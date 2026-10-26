"""Decode-path repro on the tiny MoE model: prefill correct, decode diverges?

Runs the full engine (LLM) at TP=1 on the tiny checkpoint, then replays the
first decode step manually (paged attention over the KV written by prefill)
and compares against an in-loop reference forward of the same token.

    PYTHONPATH=/root/moe-server-s4000 python3 tests/p2_decode_repro.py
"""
import sys

import torch

sys.path.insert(0, "/root/moe-server-s4000")

MODEL_DIR = "/tmp/tiny_moe"


def main():
    from nanovllm import LLM, SamplingParams

    llm = LLM(MODEL_DIR, enforce_eager=True, tensor_parallel_size=1,
              max_model_len=256, gpu_memory_utilization=0.5)
    mr = llm.model_runner
    model = mr.model

    prompt = [1, 2, 3, 4, 5, 6, 7, 8]

    # --- engine prefill step (real path, writes KV cache) -------------------
    from nanovllm.engine.sequence import Sequence
    from nanovllm.utils.context import set_context, reset_context

    seq = Sequence(prompt, SamplingParams(temperature=0, max_tokens=4, ignore_eos=True))
    seq.num_scheduled_tokens = len(prompt)
    llm.scheduler.add(seq)
    seqs, mode = llm.scheduler.schedule()
    token_ids = mr.run(seqs, mode)
    first_token = token_ids[0]
    print(f"[decode-repro] prefill next-token: {first_token}")

    # --- reference: in-loop forward of [prompt] ------------------------------
    # disable paged attention by pointing k/v caches at empty tensors
    saved = {}
    for mod in model.modules():
        if hasattr(mod, "k_cache") and hasattr(mod, "v_cache"):
            saved[id(mod)] = (mod.k_cache, mod.v_cache)
            mod.k_cache = torch.tensor([])
            mod.v_cache = torch.tensor([])
    input_ids = torch.tensor(prompt, dtype=torch.int64, device=mr.device)
    positions = torch.arange(len(prompt), device=mr.device)
    cu = torch.arange(0, len(prompt) + 1, dtype=torch.int32, device=mr.device)
    set_context(True, cu, cu, len(prompt), len(prompt),
                torch.full((len(prompt),), -1, dtype=torch.int32, device=mr.device), None, None)
    with torch.no_grad():
        h_ref = model(input_ids, positions)
        logits_ref = model.compute_logits(h_ref)
    reset_context()
    for mod in model.modules():
        if id(mod) in saved:
            mod.k_cache, mod.v_cache = saved[id(mod)]
    ref_token = logits_ref.float().argmax(-1)[-1].item()
    print(f"[decode-repro] reference next-token: {ref_token}")
    print(f"[decode-repro] prefill == reference: {first_token == ref_token}")

    # --- engine decode step (real paged path) ---------------------------------
    # append the sampled token, then run one decode step manually
    seq.append_token(first_token)
    seq.num_scheduled_tokens = 1
    seq.is_prefill = False
    from nanovllm.engine.sequence import SequenceStatus
    seq.status = SequenceStatus.RUNNING
    # rebuild block table state the way postprocess left it
    # (block_table already allocated during prefill schedule)
    decode_token = None
    # manual decode step: mirror scheduler.decode + mr.run
    llm.scheduler.running.append(seq)
    seqs, mode = llm.scheduler.schedule()
    assert mode == "decode"
    token_ids = mr.run(seqs, mode)
    decode_token = token_ids[0]
    print(f"[decode-repro] engine decode token: {decode_token}")

    # --- reference: in-loop forward of [prompt + first_token] ----------------
    for mod in model.modules():
        if hasattr(mod, "k_cache") and hasattr(mod, "v_cache"):
            saved[id(mod)] = (mod.k_cache, mod.v_cache)
            mod.k_cache = torch.tensor([])
            mod.v_cache = torch.tensor([])
    ids2 = torch.tensor(prompt + [first_token], dtype=torch.int64, device=mr.device)
    pos2 = torch.arange(len(prompt) + 1, device=mr.device)
    cu2 = torch.arange(0, len(prompt) + 2, dtype=torch.int32, device=mr.device)
    set_context(True, cu2, cu2, len(prompt) + 1, len(prompt) + 1,
                torch.full((len(prompt) + 1,), -1, dtype=torch.int32, device=mr.device), None, None)
    with torch.no_grad():
        h2 = model(ids2, pos2)
        logits2 = model.compute_logits(h2)
    reset_context()
    for mod in model.modules():
        if id(mod) in saved:
            mod.k_cache, mod.v_cache = saved[id(mod)]
    ref2 = logits2.float().argmax(-1)[-1].item()
    print(f"[decode-repro] reference decode token: {ref2}")
    print(f"[decode-repro] decode == reference: {decode_token == ref2}")

    try:
        llm.exit()
    except Exception:
        pass


if __name__ == "__main__":
    main()
