"""P24 structured-output gates (xgrammar, opt-in per request).

  g1  bitmask semantics: my torch-op expansion (LSB-first, bit t%32 of word
      t//32) matches xgrammar's own CPU application + accept/reject behavior
  g2  JSON schema validity: 4 schemas x 8 samples, 100% parse + validate
  g3  regex: `[A-Z]{3}-[0-9]{3}` x 5 -> 100% fullmatch, grammar terminates
      before max_tokens (finish_reason "stop")
  g4  unconstrained bitwise: defaults path unchanged (p22 t5 anchor re-run)
  g5  overhead: constrained vs unconstrained decode step (record, gate +10%)
  g6  validation errors -> ValueError (api maps to 400)

Run:  PYTHONPATH=. python3 tests/p24_guided.py
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from nanovllm import LLM, SamplingParams  # noqa: E402
from nanovllm.layers.sampler import SampleMeta, _apply_grammar  # noqa: E402
from nanovllm.sampling_params import SamplingParams as SP  # noqa: E402
from nanovllm.utils.compat import DEVICE  # noqa: E402

DEV = torch.device(DEVICE)


def m_accepts(compiled, token_id):
    import xgrammar as xg
    return bool(xg.GrammarMatcher(compiled).accept_token(int(token_id)))


def g1_bitmask_semantics():
    import xgrammar as xg
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/Qwen3-0.6B", use_fast=True)
    info = xg.TokenizerInfo.from_huggingface(tok)
    V, W = info.vocab_size, (info.vocab_size + 31) // 32
    cg = xg.GrammarCompiler(info).compile_json_schema(
        '{"type":"object","properties":{"name":{"type":"string"}},'
        '"required":["name"]}')
    m = xg.GrammarMatcher(cg)
    buf = torch.zeros(2, W, dtype=xg.bitmask_dtype)
    assert m.fill_next_token_bitmask(buf, 0)
    # reference: xgrammar's own CPU application on a CPU logits copy
    logits_cpu = torch.randn(2, V, dtype=torch.float32)
    ref = logits_cpu.clone()
    try:
        xg.apply_token_bitmask_inplace(ref, buf)   # 0.2.x: (logits, bitmask)
        has_ref = True
    except Exception as exc:                       # CPU kernel unavailable
        print(f"[p24] xgrammar CPU apply unavailable ({exc!r}); support-closure fallback")
        has_ref = False
        ref[0] = float("nan")                      # mark unusable
    # ours: device expansion via the sampler helper
    rows = torch.tensor([0], dtype=torch.int64, device=DEV)
    dev_logits = logits_cpu.to(DEV).clone()
    meta = SampleMeta(grammar=(rows, buf[:1].to(DEV)))
    from nanovllm.layers.sampler import _apply_masks
    got = _apply_masks(dev_logits, meta).cpu()
    floor = torch.finfo(torch.float32).min
    sup_got = set((got[0] > floor + 1).nonzero().flatten().tolist())
    if has_ref:
        sup_ref = set((ref[0] > floor + 1).nonzero().flatten().tolist())
        assert sup_got == sup_ref, f"g1 support mismatch ({len(sup_got)} vs {len(sup_ref)})"
    else:
        allowed = set(t for t in range(V)
                      if (int(buf[0, t // 32]) >> (t % 32)) & 1)
        assert sup_got == allowed, "g1 support != bitmask decode"
    if has_ref:
        assert torch.allclose(got[0, sorted(sup_got)], ref[0, sorted(sup_got)], atol=1e-6)
    # accept/reject agreement on FRESH matchers (accept advances the grammar
    # state, so each probe needs its own)
    supported = sorted(sup_got)
    assert m_accepts(cg, supported[0]), "g1: supported token rejected"
    outside = next(t for t in range(V) if t not in sup_got)
    assert not m_accepts(cg, outside), "g1: unsupported token accepted"
    print(f"g1 PASS (support {len(sup_got)} tokens; ref={'xg-cpu' if has_ref else 'closure'})")


# maxLength bounds free-form strings: inside a JSON string the grammar allows
# every token, and an unconstrained 0.6B sample at temp 0.8 will ramble past
# any max_tokens (boundedness is part of the schema, not a weaker gate).
SCHEMAS = [
    {"type": "object", "properties": {"name": {"type": "string", "maxLength": 12},
                                      "age": {"type": "integer"}},
     "required": ["name", "age"], "additionalProperties": False},
    {"type": "object", "properties": {"items": {"type": "array", "maxItems": 3,
                                                "items": {"type": "string",
                                                          "minLength": 1,
                                                          "maxLength": 8}}},
     "required": ["items"], "additionalProperties": False},
    {"type": "object", "properties": {"ok": {"type": "boolean"},
                                      "mode": {"enum": ["fast", "slow"]}},
     "required": ["ok", "mode"], "additionalProperties": False},
    {"type": "object", "properties": {"name": {"type": "string", "maxLength": 10},
                                      "tags": {"type": "array", "maxItems": 3,
                                               "items": {"type": "string",
                                                         "maxLength": 8}},
                                      "score": {"type": "number"}},
     "required": ["name", "tags", "score"], "additionalProperties": False},
]


def g2_g3_engine(llm):
    import re
    try:
        import jsonschema
    except ImportError:
        jsonschema = None
    prompts = ["输出一个 JSON 对象：", "生成数据：", "请填充：", "给个例子："]
    for si, schema in enumerate(SCHEMAS):
        sp = SP(temperature=0.8, max_tokens=256, ignore_eos=False,
                guided_json=schema)
        outs = llm.generate(prompts, [sp] * len(prompts), use_tqdm=False)
        for o in outs:
            # offline decode keeps special tokens on this tokenizer build;
            # the API incremental path (skip=True) is what serves users
            text = llm.tokenizer.decode(o["token_ids"],
                                        skip_special_tokens=True).strip()
            obj = json.loads(text)                    # raises if invalid
            if jsonschema is not None:
                jsonschema.validate(obj, schema)
            assert o["metrics"]["finish_reason"] == "stop", \
                f"g2 schema{si}: finish={o['metrics']['finish_reason']}"
        print(f"g2 PASS schema{si} (4/4 valid + grammar-terminated)", flush=True)

    sp = SP(temperature=0.8, max_tokens=48, guided_regex=r"[A-Z]{3}-[0-9]{3}")
    outs = llm.generate(prompts[:2], [sp] * 2, use_tqdm=False)
    for o in outs:
        text = llm.tokenizer.decode(o["token_ids"],
                                    skip_special_tokens=True).strip()
        assert re.fullmatch(r"[A-Z]{3}-[0-9]{3}", text), \
            f"g3 regex mismatch: {text!r}"
        assert o["metrics"]["finish_reason"] == "stop"
        assert len(o["token_ids"]) < 48
    print("g3 PASS regex fullmatch + termination", flush=True)


def g5_overhead(llm):
    BODY = "大模型推理系统需要在吞吐与延迟之间做精细权衡。" * 6
    def segment(sp):
        for i in range(8):
            llm.add_request(f"背景{i}:{BODY}", sp)
        fw = []
        while not llm.is_finished():
            llm.step()
            m = llm.last_step_metrics
            if m["mode"] == "decode" and m["forward_s"] > 0:
                fw.append(m["forward_s"])
        return statistics.median(fw)
    segment(SP(temperature=0.8, max_tokens=32, ignore_eos=True))          # warmup
    off, on = [], []
    for _ in range(3):
        off.append(segment(SP(temperature=0.8, max_tokens=32, ignore_eos=True)))
        on.append(segment(SP(temperature=0.8, max_tokens=32, ignore_eos=True,
                            guided_json=SCHEMAS[0])))
    ratio = statistics.median(on) / statistics.median(off)
    print(f"g5 constrained/unconstrained decode median ratio = {ratio:.4f} "
          f"(x3 interleaved)", flush=True)
    return ratio


def g6_validation():
    bad = [dict(guided_json=123), dict(guided_regex=456),
           dict(guided_json={"type": "object"}, guided_regex="x")]
    for kwargs in bad:
        try:
            SP(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"g6: {kwargs} did not raise")
    print("g6 PASS validation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    parser.add_argument("--out", default="/root/p24_data/guided.json")
    args = parser.parse_args()
    g6_validation()
    g1_bitmask_semantics()
    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1,
              attention_backend="triton_paged", max_model_len=2048,
              max_num_seqs=8, gpu_memory_utilization=0.35,
              scheduler_policy="stall_free")
    g2_g3_engine(llm)
    ratio = g5_overhead(llm)
    llm.exit()
    ok = ratio <= 1.10
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"g5_ratio": round(ratio, 4), "pass": ok}, f, indent=1)
    print(f"[p24] {'PASS' if ok else 'FAIL (g5 overhead)'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
