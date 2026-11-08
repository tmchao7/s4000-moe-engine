"""P22 bug-fix regression: the three real bugs caught during P22-A, each with
a dedicated test (failing before the fix, passing after).

  b1  _run_mixed logprobs=None crash — mixed steps without logprobs crashed on
      lp[row] ('NoneType' not subscriptable).  Forces the legacy split path
      via NANOVLLM_MIXED_SPLIT=1 (env must precede any nanovllm import),
      drives a mixed batch (decode + mid-flight prefill): must complete with
      no logprobs; with logprobs=3 every sampled row carries an lp dict.
  b2  IncrementalDetokenizer.__init__ missing skip_special_tokens kwarg —
      TypeError at the API layer.  False keeps special tokens, True (default)
      skips them.
  b3  sampler mutated fp32 caller logits (float() aliases fp32 input, then
      div_/penalty-scatter write in place) — the meta path must leave the
      input tensor bitwise unchanged.

Run:  PYTHONPATH=. python3 tests/p22_bugfix_regress.py [--model PATH]
"""
import os
os.environ.setdefault("NANOVLLM_MIXED_SPLIT", "1")   # b1: force the split path

import argparse
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from nanovllm.layers.sampler import Sampler, SampleMeta  # noqa: E402
from nanovllm.utils.compat import DEVICE, IS_MUSA  # noqa: E402

DEV = torch.device(DEVICE)


def seed_all(seed):
    torch.manual_seed(seed)
    if IS_MUSA:
        torch.musa.manual_seed(seed)


def b3_sampler_purity():
    """Meta path (penalty + masks + logprobs, greedy and sampled) must not
    mutate the caller's fp32 logits."""
    seed_all(7)
    sampler = Sampler()
    vocab = 512
    logits = torch.randn(4, vocab, dtype=torch.float32, device=DEV)
    snapshot = logits.clone()
    temps = torch.tensor([0.0, 0.8, 0.0, 0.9], dtype=torch.float32, device=DEV)
    meta = SampleMeta(
        penalty=(
            torch.tensor([0, 1, 1, 2, 3], dtype=torch.int64, device=DEV),
            torch.tensor([10, 11, 12, 11, 13], dtype=torch.int64, device=DEV),
            torch.tensor([1, 2, 1, 3, 1], dtype=torch.int32, device=DEV),
            torch.tensor([0.2] * 5, dtype=torch.float32, device=DEV),
            torch.tensor([0.3] * 5, dtype=torch.float32, device=DEV),
            torch.tensor([1.1] * 5, dtype=torch.float32, device=DEV),
        ),
        top_ps=torch.tensor([1.0, 0.9, 1.0, 0.8], dtype=torch.float32, device=DEV),
        top_ks=torch.tensor([-1, 50, -1, 20], dtype=torch.int32, device=DEV),
        min_ps=None, has_k=True, has_p=True, has_minp=False, logprobs_k=3,
    )
    for all_greedy in (None, True, False):
        sampler(logits, temps, all_greedy, meta)
        assert torch.equal(logits, snapshot), f"b3: input mutated (all_greedy={all_greedy})"
    print("PASS b3 sampler purity (fp32 input bitwise unchanged, 3 modes)")


def b2_detokenizer_skip(model):
    from transformers import AutoTokenizer
    from nanovllm.incremental_detokenizer import IncrementalDetokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    eos = tokenizer.eos_token_id
    assert eos is not None
    kept = IncrementalDetokenizer(tokenizer, skip_special_tokens=False)
    delta_kept = kept.push(eos)
    skipped = IncrementalDetokenizer(tokenizer)   # default True
    delta_skipped = skipped.push(eos)
    assert delta_skipped == "", f"b2: default should skip specials, got {delta_skipped!r}"
    assert delta_kept != "", "b2: skip_special_tokens=False emitted no text"
    print(f"PASS b2 detokenizer skip_special_tokens (kept {delta_kept!r} / skipped {delta_skipped!r})")


def b1_mixed_split(model):
    """Mixed steps through the legacy split path: no-logprobs must complete
    (exact pre-fix crash), logprobs=3 must populate every sampled row."""
    from nanovllm import LLM, SamplingParams
    llm = LLM(model, enforce_eager=True, tensor_parallel_size=1,
              attention_backend="triton_paged", max_model_len=2048,
              max_num_seqs=8, gpu_memory_utilization=0.35,
              scheduler_policy="stall_free")
    body = "大模型推理系统需要在吞吐与延迟之间做精细权衡，prefill 计算密集而 decode 访存密集。" * 4

    for tag, logprobs in (("no-logprobs", None), ("with-logprobs", 3)):
        sp = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True,
                            logprobs=logprobs)
        # two decodes first, then a fresh prefill joins mid-decode -> mixed
        sids = [llm.add_request(f"背景{i}:{body}", sp) for i in (0, 1)]
        events = []
        mixed_seen = False
        sid_new = None
        stepped = 0
        while not llm.is_finished():
            if stepped == 2:
                sid_new = llm.add_request(f"插队:{body}", sp)
                sids.append(sid_new)
            outputs, _ = llm.step()
            stepped += 1
            if llm.last_step_metrics["mode"] == "mixed":
                mixed_seen = True
            events += llm.last_token_events
        assert mixed_seen, f"b1[{tag}]: no mixed step observed"
        assert all(seq in [e[0] for e in events] for seq in sids), \
            f"b1[{tag}]: missing finished seq events"
        if logprobs:
            n_lp = sum(1 for e in events if e[4] is not None)
            n_tok = sum(len([e for e in events if e[0] == s]) for s in sids)
            assert n_lp == n_tok and n_lp >= 3 * len(sids), \
                f"b1[{tag}]: lp rows {n_lp} != token rows {n_tok}"
        print(f"PASS b1 mixed-split {tag} (mixed_seen, {len(events)} events)")
    llm.exit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    args = parser.parse_args()
    b3_sampler_purity()
    b2_detokenizer_skip(args.model)
    b1_mixed_split(args.model)
    print("[p22-bugfix] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
