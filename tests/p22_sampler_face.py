"""P22-A sampling-face tests.

Pure-torch tests (t0-t4) need only the accelerator; --e2e adds engine tests
t5-t7 on 0.6B (t5 diffs greedy output against the pre-change capture from
tests/p22_baseline_greedy.py — the bitwise no-op gate).

  t0  SamplingParams validation (ValueError on bad ranges)
  t1  mask/penalty transforms vs independently-written CPU reference
      (support-set equality + allclose + argmax equality, margin-designed
      logits so no float-boundary straddling)
  t2  support closure: masked-out tokens are never sampled (200 draws)
  t3  penalty arithmetic: hand-worked example, elementwise exact
  t4  no-op equivalence: all-default meta (and meta=None) reproduce the
      verbatim legacy sampler bitwise under the same RNG seed
  t5  0.6B defaults greedy == pre-change baseline capture (bitwise)
  t6  0.6B per-knob sanity + engine-level stop_token_ids + logprobs shape
      (greedy ⇒ top[0] id == emitted token id)
  t7  perf ABA: default vs top_p=0.9 decode forward_s (gate: knobs ≤ +10%)

Run:  PYTHONPATH=. python3 tests/p22_sampler_face.py           # t0-t4
      PYTHONPATH=. python3 tests/p22_sampler_face.py --e2e     # t0-t7
"""
import argparse
import json
import math
import os
import statistics
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))

from p22_baseline_greedy import PROMPTS  # noqa: E402  (single source of prompts)
from nanovllm.layers.sampler import Sampler, SampleMeta, _apply_masks, _apply_penalties  # noqa: E402
from nanovllm.sampling_params import SamplingParams  # noqa: E402
from nanovllm.utils.compat import DEVICE, IS_MUSA  # noqa: E402

DEV = torch.device(DEVICE)
BASELINE = "/root/p22_data/baseline_greedy.json"
FLOOR = torch.finfo(torch.float32).min
# masked positions hold exactly finfo.min; real logits live near [-20, 20]
_UNMASKED = -1e30


def seed_all(seed):
    torch.manual_seed(seed)
    if IS_MUSA:
        torch.musa.manual_seed(seed)


def support(x):
    return set((x > _UNMASKED).nonzero().flatten().tolist())


def meta_knobs(top_k=-1, top_p=1.0, min_p=0.0, rows=1):
    return SampleMeta(
        top_ps=torch.tensor([top_p] * rows, dtype=torch.float32, device=DEV) if top_p < 1.0 else None,
        top_ks=torch.tensor([top_k] * rows, dtype=torch.int32, device=DEV) if top_k != -1 else None,
        min_ps=torch.tensor([min_p] * rows, dtype=torch.float32, device=DEV) if min_p > 0.0 else None,
        has_k=top_k != -1, has_p=top_p < 1.0, has_minp=min_p > 0.0,
    )


# --------------------------------------------------------------------- t0

def t0_validation():
    bad = [
        dict(top_p=0.0), dict(top_p=1.5), dict(top_k=0), dict(top_k=-2),
        dict(min_p=-0.1), dict(presence_penalty=3.0), dict(frequency_penalty=-2.5),
        dict(repetition_penalty=0.0), dict(logprobs=21), dict(logprobs=-1),
        dict(stop_token_ids=["x"]), dict(stop_token_ids=[True]),
    ]
    for kwargs in bad:
        try:
            SamplingParams(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"t0: {kwargs} did not raise ValueError")
    SamplingParams(top_k=-1, logprobs=0, stop_token_ids=[1, 2, 3])  # ok
    print("PASS t0 param validation")


# ------------------------------------------------------ reference impls

def ref_masks(logits, top_k=-1, top_p=1.0, min_p=0.0):
    """Independent CPU reference (float32) for the mask pipeline."""
    x = logits.clone()
    if top_k != -1:
        kth = x.sort(dim=-1, descending=True).values[:, top_k - 1:top_k]
        x = x.masked_fill(x < kth, FLOOR)
    if top_p < 1.0:
        s, si = x.sort(dim=-1, descending=True)
        p = torch.softmax(s, dim=-1)
        remove = (p.cumsum(dim=-1) - p) > top_p
        remove[:, 0] = False
        x = torch.empty_like(x).scatter(1, si, s.masked_fill(remove, FLOOR))
    if min_p > 0.0:
        p = torch.softmax(x, dim=-1)
        x = x.masked_fill(p < min_p * p.max(dim=-1, keepdim=True).values, FLOOR)
    return x


def ref_penalties(logits, appeared_prompt, appeared_gen, rep, presence, freq):
    """Independent CPU reference.  repetition sees prompt+generated;
    presence/frequency only the generated counts."""
    x = logits.clone()
    for tid in set(appeared_prompt) | set(appeared_gen):
        v = x[0, tid]
        v = v * rep if v < 0 else v / rep
        if tid in appeared_gen:
            v = v - presence - freq * appeared_gen[tid]
        x[0, tid] = v
    return x


# --------------------------------------------------------------------- t1

def t1_masks_reference():
    seed_all(11)
    logits = torch.randn(4, 512, dtype=torch.float32)
    # widen margins so no cumsum/exp boundary straddles a ulp
    logits[:, 7] += 3.0
    logits[:, 3] += 2.5
    logits[:, 29] += 2.0
    dev_logits = logits.to(DEV)
    cases = [
        dict(top_k=5), dict(top_k=1), dict(top_p=0.5), dict(top_p=0.9),
        dict(min_p=0.2), dict(top_k=10, top_p=0.7), dict(top_k=3, min_p=0.25),
        dict(top_p=0.6, min_p=0.15), dict(top_k=8, top_p=0.55, min_p=0.1),
        dict(top_k=-1, top_p=1.0, min_p=0.0),   # neutral: no-op
    ]
    for i, case in enumerate(cases):
        row = i % 4
        m = meta_knobs(**case)
        got = _apply_masks(dev_logits[row:row + 1].clone(), m).cpu()
        want = ref_masks(logits[row:row + 1], **case)
        assert support(got) == support(want), f"t1 support mismatch {case}"
        assert torch.allclose(got, want, atol=1e-5), f"t1 values mismatch {case}"
        assert got.argmax() == want.argmax(), f"t1 argmax mismatch {case}"
    print(f"PASS t1 masks vs CPU reference ({len(cases)} cases)")


def t1b_penalty_reference():
    # rep applies to prompt+generated; presence/frequency only to generated
    logits = torch.tensor([[-1.0, 2.0, 0.5, -0.5, 1.5, -2.0, 3.0, 0.1]], dtype=torch.float32)
    gen, prompt = {1: 3, 4: 1}, {5: 1, 1: 3, 4: 1}
    rows = torch.tensor([0, 0, 0, 0], dtype=torch.int64, device=DEV)
    ids = torch.tensor([1, 4, 5, 1], dtype=torch.int64, device=DEV)
    counts = torch.tensor([3, 1, 1, 3], dtype=torch.int32, device=DEV)
    pre = torch.tensor([0.25, 0.25, 0.0, 0.25], dtype=torch.float32, device=DEV)
    fre = torch.tensor([0.5, 0.5, 0.0, 0.5], dtype=torch.float32, device=DEV)
    rep = torch.tensor([1.5] * 4, dtype=torch.float32, device=DEV)
    got = _apply_penalties(logits.to(DEV).clone(), (rows, ids, counts, pre, fre, rep)).cpu()
    want = ref_penalties(logits, prompt, gen, 1.5, 0.25, 0.5)
    assert torch.equal(got, want), f"t1b penalty mismatch:\n{got}\n{want}"
    print("PASS t1b penalties vs CPU reference (bitwise)")


# --------------------------------------------------------------------- t2

def t2_support_closure():
    seed_all(22)
    logits = torch.randn(1, 1000, dtype=torch.float32, device=DEV)
    logits[0, :20] += 4.0                      # strong head so the nucleus is small
    sampler = Sampler()
    temp_value = 1.5
    temp = torch.tensor([temp_value], dtype=torch.float32, device=DEV)
    cases = [dict(top_k=10), dict(top_p=0.3), dict(min_p=0.2), dict(top_k=7, top_p=0.4)]
    for case in cases:
        meta = meta_knobs(**case)
        # mirror the sampler order: temperature scaling happens BEFORE masks
        scaled = logits / temp_value
        allowed = support(_apply_masks(scaled.clone(), meta))
        assert len(allowed) >= 1
        for _ in range(200):
            tokens, _lp = sampler(logits, temp, False, meta)
            assert tokens.item() in allowed, f"t2: token {tokens.item()} outside support {case}"
    print(f"PASS t2 support closure ({len(cases)} knobs x 200 draws)")


# --------------------------------------------------------------------- t3

def t3_penalty_hand_example():
    # hand-worked: token 3 (generated x2), token 6 (prompt only)
    logits = torch.tensor([[0.0, 1.0, -2.0, 4.0, 0.25, -0.75, 8.0, -3.0]],
                          dtype=torch.float32)
    rows = torch.tensor([0, 0], dtype=torch.int64, device=DEV)
    ids = torch.tensor([3, 6], dtype=torch.int64, device=DEV)
    counts = torch.tensor([2, 1], dtype=torch.int32, device=DEV)
    pre = torch.tensor([0.5, 0.0], dtype=torch.float32, device=DEV)
    fre = torch.tensor([1.0, 0.0], dtype=torch.float32, device=DEV)
    rep = torch.tensor([2.0] * 2, dtype=torch.float32, device=DEV)
    got = _apply_penalties(logits.to(DEV).clone(), (rows, ids, counts, pre, fre, rep)).cpu()
    # token 3: 4.0 / 2.0 - 0.5 - 1.0*2 = -0.5 ; token 6: 8.0 / 2.0 = 4.0
    want = torch.tensor([[0.0, 1.0, -2.0, -0.5, 0.25, -0.75, 4.0, -3.0]], dtype=torch.float32)
    assert torch.equal(got, want), f"t3 mismatch:\n{got}\n{want}"
    print("PASS t3 penalty hand example (bitwise)")


# --------------------------------------------------------------------- t4

def _legacy_forward(logits, temperatures, all_greedy):
    """The verbatim pre-P22 sampler body (the no-op oracle)."""
    greedy = temperatures < 1e-3
    if all_greedy is None:
        all_greedy = bool(greedy.all())
    if all_greedy:
        return logits.float().argmax(dim=-1)
    logits = logits.float().div_(temperatures.clamp_min(1e-10).unsqueeze(dim=1))
    probs = torch.softmax(logits, dim=-1)
    sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
    return torch.where(greedy, logits.argmax(dim=-1), sample_tokens)


def t4_noop_equivalence():
    seed_all(44)
    sampler = Sampler()
    logits = torch.randn(8, 4096, dtype=torch.float32, device=DEV)
    logits[:, 100] += 5.0                       # deterministic greedy rows
    temps = torch.tensor([0.0, 0.0, 0.7, 0.7, 1.1, 0.0, 0.9, 1.3],
                         dtype=torch.float32, device=DEV)
    for all_greedy in (True, False, None):
        seed_all(1234)
        legacy = _legacy_forward(logits.clone(), temps, all_greedy)
        seed_all(1234)
        new_a, lp_a = sampler(logits.clone(), temps, all_greedy, None)
        seed_all(1234)
        neutral = SampleMeta()                  # constructed-but-neutral
        new_b, lp_b = sampler(logits.clone(), temps, all_greedy, neutral)
        assert torch.equal(legacy, new_a), f"t4 meta=None mismatch (all_greedy={all_greedy})"
        assert torch.equal(legacy, new_b), f"t4 neutral-meta mismatch (all_greedy={all_greedy})"
        assert lp_a is None and lp_b is None
    assert SampleMeta().is_neutral
    assert not SampleMeta(logprobs_k=1).is_neutral
    print("PASS t4 no-op equivalence vs legacy body (bitwise, 3 modes)")


# ----------------------------------------------------------------- e2e

def e2e(model_path):
    from nanovllm import LLM, SamplingParams
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1,
              attention_backend="triton_paged", max_model_len=2048,
              max_num_seqs=32, gpu_memory_utilization=0.35,
              scheduler_policy="stall_free")
    try:
        return _e2e_body(llm)
    finally:
        llm.exit()


def _e2e_body(llm):
    # t5: defaults greedy == pre-change capture (bitwise no-op gate)
    sp = SamplingParams(temperature=0, max_tokens=48, ignore_eos=True)
    outs = llm.generate(PROMPTS, [sp] * len(PROMPTS), use_tqdm=False)
    tokens = [list(o["token_ids"]) for o in outs]
    assert os.path.exists(BASELINE), f"t5: baseline capture missing at {BASELINE}"
    with open(BASELINE) as f:
        base = json.load(f)
    assert base["tokens"] == tokens, "t5 FAIL: defaults greedy diverged from pre-change baseline"
    print("PASS t5 defaults greedy == pre-change baseline (bitwise)")

    # t6a: per-knob sanity — finishes, 32 tokens, no crash/garbage
    knobs = [
        dict(temperature=0.8, top_p=0.9),
        dict(temperature=0.8, top_k=50),
        dict(temperature=0.8, min_p=0.1),
        dict(temperature=0.8, presence_penalty=0.3),
        dict(temperature=0.8, frequency_penalty=0.4),
        dict(temperature=0.8, repetition_penalty=1.2),
        dict(temperature=0.8, top_p=0.9, top_k=50, min_p=0.1,
             presence_penalty=0.2, frequency_penalty=0.3, repetition_penalty=1.1),
    ]
    for case in knobs:
        sp = SamplingParams(max_tokens=32, ignore_eos=True, **case)
        outs = llm.generate(PROMPTS[:2], [sp] * 2, use_tqdm=False)
        for o in outs:
            assert len(o["token_ids"]) == 32, f"t6a short output {case}"
    print(f"PASS t6a per-knob sanity ({len(knobs)} cases x 2 prompts)")

    # t6b: engine-level stop_token_ids — deterministic: stop at the 2nd token
    # of a reference greedy generation
    sp = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
    ref = llm.generate(PROMPTS[:1], sp, use_tqdm=False)[0]["token_ids"]
    sp_stop = SamplingParams(temperature=0, max_tokens=16,
                             stop_token_ids=[ref[1]])
    out = llm.generate(PROMPTS[:1], sp_stop, use_tqdm=False)[0]
    assert out["metrics"]["finish_reason"] == "stop", \
        f"t6b finish_reason={out['metrics']['finish_reason']}"
    assert len(out["token_ids"]) == 2, f"t6b stopped after {len(out['token_ids'])} tokens, want 2"
    print("PASS t6b engine-level stop_token_ids (deterministic 2-token stop)")

    # t6c: logprobs — greedy ⇒ chosen is the argmax ⇒ top[0] id == token id
    sp = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True, logprobs=5)
    sid = llm.add_request(PROMPTS[0], sp)
    events = []
    while not llm.is_finished():
        llm.step()
        events += llm.last_token_events
    n_lp = 0
    for seq_id, token_id, finished, metrics, lp in events:
        if seq_id != sid or lp is None:
            continue
        n_lp += 1
        assert len(lp["top"]) == 5 and all(v <= 1e-6 for _, v in lp["top"])
        assert lp["top"][0][0] == token_id, f"t6c top1 {lp['top'][0][0]} != token {token_id}"
        assert math.isfinite(lp["chosen"]) and lp["chosen"] <= 1e-6
    assert n_lp == 8, f"t6c expected 8 logprob events, got {n_lp}"
    print("PASS t6c logprobs shape + greedy top1==chosen")

    # t7: perf ABA — default (neutral path) vs top_p=0.9 (sort pipeline)
    def segment(temperature=0.0, top_p=1.0):
        sp = SamplingParams(temperature=temperature, max_tokens=48,
                            ignore_eos=True, top_p=top_p)
        for i in range(16):
            llm.add_request(f"背景{i}: 大模型推理系统需要在吞吐与延迟之间做精细权衡。" * 4, sp)
        fw = []
        while not llm.is_finished():
            llm.step()
            m = llm.last_step_metrics
            if m["mode"] == "decode":
                fw.append(m["forward_s"])
        return statistics.median(fw)

    segment()                                    # warmup
    base_fw, knob_fw = [], []
    for _ in range(3):
        base_fw.append(segment())
        knob_fw.append(segment(temperature=0.9, top_p=0.9))
    b = statistics.median(base_fw)
    k = statistics.median(knob_fw)
    ratio = k / b
    print(f"t7 decode median: default {b * 1e3:.2f}ms top_p0.9 {k * 1e3:.2f}ms "
          f"ratio={ratio:.4f} (interleaved x3)")
    return ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e", action="store_true")
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    args = parser.parse_args()
    t0_validation()
    t1_masks_reference()
    t1b_penalty_reference()
    t2_support_closure()
    t3_penalty_hand_example()
    t4_noop_equivalence()
    if not args.e2e:
        print("[p22-sampler-face] unit tests PASS")
        return 0
    ratio = e2e(args.model)
    ok = ratio <= 1.10
    print(f"[p22-sampler-face] e2e {'PASS' if ok else 'FAIL (t7 knob overhead)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
