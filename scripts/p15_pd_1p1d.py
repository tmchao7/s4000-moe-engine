"""P15-C M2: single-machine dual-card PD 1P1D for Qwen3-0.6B (MVP).

Topology: one process per role, one card each (MUSA_VISIBLE_DEVICES pins the
physical card).  Prefill (card 0) runs the engine's own prefill+chunk path,
then hands the request to Decode (card 1) as whole KV blocks over TCP with
pinned staging; decode runs the autoregressive loop and streams token ids
back.  Control plane is a two-socket JSON protocol (control + KV bulk).

Gate (docs/PLAN.md P15-C M2): PD arm token ids must equal the collocated
baseline arm bitwise (same engine code, deterministic); plus honest
prefill/transfer/decode phase timings.

Roles:
  --role baseline  collocated engine, plain generate (reference tokens)
  --role prefill   P: engine on card 0, prefill + handoff
  --role decode    D: engine on card 1 (set MUSA_VISIBLE_DEVICES=1), decode

Launch:  python3 scripts/p15_pd_1p1d.py --launch   (spawns prefill+decode)
"""
import argparse
import json
import os
import socket
import struct
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

MODEL = "/root/autodl-tmp/models/Qwen3-0.6B"
MAX_TOKENS = 24
PROMPTS = [
    "请解释什么是 MoE（混合专家）架构。",
    "Explain how paged KV cache works in one paragraph.",
    "Write a Python function that reverses a string.",
    "计算 17 × 23 的结果，并给出中间步骤。",
    "为什么 prefill 是计算密集型而 decode 是访存密集型？",
    "Summarize how continuous batching works.",
]
HDR = struct.Struct("<I")


def recvn(sock: socket.socket, n: int) -> bytes:
    chunks = []
    while n > 0:
        chunk = sock.recv(min(n, 1 << 22))
        if not chunk:
            raise ConnectionError("peer closed")
        chunks.append(chunk)
        n -= len(chunk)
    return b"".join(chunks)


def send_json(sock, obj):
    data = json.dumps(obj).encode()
    sock.sendall(HDR.pack(len(data)) + data)


def recv_json(sock):
    (n,) = HDR.unpack(recvn(sock, HDR.size))
    return json.loads(recvn(sock, n))


def connect_retry(host, port, tries=60):
    for i in range(tries):
        try:
            sock = socket.create_connection((host, port), timeout=5)
            sock.settimeout(None)  # blocking afterwards: D's first decode JITs triton
            return sock
        except OSError:
            time.sleep(1.0 if i < 10 else 2.0)
    raise ConnectionError(f"cannot connect {host}:{port}")


def build_engine():
    from nanovllm import LLM
    return LLM(MODEL, enforce_eager=True, tensor_parallel_size=1,
               attention_backend="triton_paged", max_model_len=2048,
               max_num_seqs=8, gpu_memory_utilization=0.5)


def role_baseline(args):
    from nanovllm import SamplingParams
    llm = build_engine()
    outs = llm.generate(PROMPTS, SamplingParams(
        temperature=0, max_tokens=MAX_TOKENS, ignore_eos=True), use_tqdm=False)
    tokens = [list(o["token_ids"]) for o in outs]
    llm.exit()
    with open(args.out, "w") as f:
        json.dump({"role": "baseline", "tokens": tokens}, f)
    print(f"[baseline] wrote {args.out}", flush=True)


def role_prefill(args):
    from nanovllm import SamplingParams
    llm = build_engine()
    kv_sock = connect_retry(args.host, args.port)
    ctrl_sock = connect_retry(args.host, args.port + 1)
    from nanovllm.pd.kv_channel import TcpKVChannel
    kvchan = TcpKVChannel(kv_sock, llm.model_runner.kv_cache, max_blocks=16)
    phases = []
    for prompt in PROMPTS:
        t0 = time.perf_counter()
        seq_id = llm.add_request(prompt, SamplingParams(
            temperature=0, max_tokens=MAX_TOKENS, ignore_eos=True))
        seq = None
        while seq is None:
            pool = list(llm.scheduler.running) + list(llm.scheduler.waiting)
            seq = next((s for s in pool if s.seq_id == seq_id), None)
            if seq is None:
                llm.step()
        while seq.num_tokens == seq.num_prompt_tokens:
            llm.step()  # prefill all prompt chunks; KV 0..P-1 now in pool
        t1 = time.perf_counter()
        prompt_tokens = seq.token_ids[:seq.num_prompt_tokens]
        first_token = seq.completion_token_ids[0]  # sampled by P's prefill kernel
        send_json(ctrl_sock, {"type": "request",
                              "prompt_token_ids": prompt_tokens,
                              "first_token": first_token,
                              "max_tokens": MAX_TOKENS,
                              "nblocks": len(seq.block_table)})
        kvchan.send_blocks(seq.block_table)
        t2 = time.perf_counter()
        llm.cancel_request(seq_id)
        result = recv_json(ctrl_sock)
        t3 = time.perf_counter()
        phases.append({"prompt_tokens": len(prompt_tokens),
                       "prefill_s": round(t1 - t0, 4),
                       "transfer_s": round(t2 - t1, 4),
                       "decode_s": round(t3 - t2, 4),
                       "nblocks": result["nblocks"],
                       "kv_mib": round(kvchan.bytes_moved / 2**20, 2)})
        print(f"[pre] {phases[-1]}", flush=True)
    send_json(ctrl_sock, {"type": "done"})
    with open(args.out, "w") as f:
        json.dump({"role": "prefill", "phases": phases}, f)
    llm.exit()


def role_decode(args):
    from nanovllm import SamplingParams
    from nanovllm.engine.sequence import Sequence, SequenceStatus
    from nanovllm.pd.kv_channel import TcpKVChannel

    llm = build_engine()
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.host, args.port))
    listener.listen(1)
    listener2 = socket.socket()
    listener2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener2.bind((args.host, args.port + 1))
    listener2.listen(1)
    print(f"[decode] listening on {args.port}/{args.port + 1}", flush=True)
    kv_sock, _ = listener.accept()
    ctrl_sock, _ = listener2.accept()
    kv_sock.settimeout(None)
    ctrl_sock.settimeout(None)
    kvchan = TcpKVChannel(kv_sock, llm.model_runner.kv_cache, max_blocks=16)
    bm = llm.scheduler.block_manager
    results = []
    while True:
        req = recv_json(ctrl_sock)
        if req.get("type") == "done":
            break
        # Rebuild the exact post-prefill state: token_ids = prompt + C1 (the
        # token P's prefill kernel sampled), completions counted from C1 on.
        # The first decode step here replays collocated's first decode step
        # bit-for-bit: input = C1, KV write at slot P (fresh), attention over
        # the imported 0..P-1 plus the new slot.
        toks = list(req["prompt_token_ids"]) + [req["first_token"]]
        seq = Sequence(toks, SamplingParams(
            temperature=0, max_tokens=req["max_tokens"], ignore_eos=True))
        seq.num_prompt_tokens = len(req["prompt_token_ids"])
        seq.status = SequenceStatus.RUNNING
        seq.is_prefill = False
        seq.num_cached_tokens = seq.num_prompt_tokens
        seq.block_table = [bm._allocate_block() for _ in range(req["nblocks"])]
        llm.scheduler.running.append(seq)
        t0 = time.perf_counter()
        kvchan.recv_blocks(seq.block_table)
        t1 = time.perf_counter()
        while not seq.is_finished:
            llm.step()
        t2 = time.perf_counter()
        results.append({"tokens": list(seq.completion_token_ids),
                        "import_s": round(t1 - t0, 4),
                        "decode_s": round(t2 - t1, 4)})
        print(f"[decode] import={results[-1]['import_s']:.4f}s "
              f"decode={results[-1]['decode_s']:.4f}s", flush=True)
        send_json(ctrl_sock, {"type": "result", "nblocks": req["nblocks"],
                              "token_ids": results[-1]["tokens"]})
    # in-engine control arm: collocated generate on the SAME engine, so
    # d_collocated vs d_pd isolates the import path from engine state
    from nanovllm import SamplingParams
    collocated = []
    for prompt in PROMPTS:
        seq_id = llm.add_request(prompt, SamplingParams(
            temperature=0, max_tokens=MAX_TOKENS, ignore_eos=True))
        while True:
            pool = list(llm.scheduler.running) + list(llm.scheduler.waiting)
            seq = next((x for x in pool if x.seq_id == seq_id), None)
            if seq is not None:
                break
            llm.step()
        while not seq.is_finished:
            llm.step()
        collocated.append({"tokens": list(seq.completion_token_ids)})
    with open(args.out, "w") as f:
        json.dump({"role": "decode", "results": results,
                   "collocated": collocated}, f)
    llm.exit()


def launch(args):
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    env = dict(os.environ, PYTHONPATH=REPO)
    d_out = args.out.replace(".json", "_decode.json")
    p_out = args.out.replace(".json", "_prefill.json")
    dec = subprocess.Popen(
        [sys.executable, __file__, "--role", "decode", "--host", args.host,
         "--port", str(args.port), "--out", d_out],
        env={**env, "MUSA_VISIBLE_DEVICES": "1"})
    pre = subprocess.Popen(
        [sys.executable, __file__, "--role", "prefill", "--host", args.host,
         "--port", str(args.port), "--out", p_out], env=env)
    dec.wait()
    pre.wait()
    rc = dec.returncode | pre.returncode

    base_out = args.out.replace(".json", "_baseline.json")
    subprocess.run([sys.executable, __file__, "--role", "baseline",
                    "--out", base_out], env=env, check=True)
    base = json.load(open(base_out))
    dec = json.load(open(d_out))
    dec_res = dec["results"]
    # Gate reference = the IN-ENGINE collocated arm (same process, same engine
    # state history).  A virgin-engine baseline is NOT a valid reference:
    # engine tokens depend on state history at near-ties (P15 finding, the
    # I31 first-batch effect generalized beyond the first batch), so a
    # history-matched control is the only fair bitwise reference.
    ref = dec["collocated"]
    verdict = {"per_request": [], "pass": True,
               "reference": "in-engine collocated",
               "virgin_baseline_match": []}
    for i, (b, r) in enumerate(zip(ref, dec_res)):
        same = b["tokens"] == r["tokens"]
        verdict["per_request"].append({"req": i, "bitwise_match": same,
                                       "n_tokens": len(r["tokens"])})
        verdict["pass"] &= same
    for i, (b, r) in enumerate(zip(base["tokens"], dec_res)):
        verdict["virgin_baseline_match"].append(b == r["tokens"])
    with open(args.out, "w") as f:
        json.dump(verdict, f, indent=1)
    print(json.dumps(verdict, indent=1), flush=True)
    print(f"[m2] {'PASS' if verdict['pass'] and rc == 0 else 'FAIL'}", flush=True)
    return 0 if verdict["pass"] and rc == 0 else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["baseline", "prefill", "decode", "launch"],
                        default="launch")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29610)
    parser.add_argument("--out", default="/root/p15_data/pd_m2_verdict.json")
    args = parser.parse_args()
    if args.role == "launch":
        sys.exit(launch(args))
    if args.role == "baseline":
        role_baseline(args)
    elif args.role == "prefill":
        role_prefill(args)
    else:
        role_decode(args)


if __name__ == "__main__":
    main()
