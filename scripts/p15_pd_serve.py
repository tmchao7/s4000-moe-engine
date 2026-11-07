"""P15-C-M3: concurrent PD serving roles + collocated server (same boundary).

Roles (all enforce_eager — no graph capture anywhere):
  collocated  TP2 engine across both cards, framed client server; the A/B
              reference arm — same hardware, same workload, standard engine.
  prefill     P: engine on card 0, client-facing router; batches prefills,
              hands each request to D as soon as its prefill completes
              (ctrl frame first, then the KV bulk frame — order preserved).
  decode      D: engine on card 1; admits handoffs between decode steps
              (non-blocking KV frame assembly), decodes the whole batch —
              the engine's own batching provides the concurrency.

Protocol: every JSON socket carries [u32 length][payload] frames; the P<->D
KV socket carries TcpKVChannel bulk frames, reassembled incrementally so
decode steps never stall behind a transfer.
"""
import argparse
import json
import selectors
import socket
import struct
import sys
import time
import os
from collections import deque

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from nanovllm.pd.protocol import (  # noqa: F401
    HDR, JsonStream, connect_retry, recvn_blocking, send_json,
)


def listen(port):
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    s.listen(4)
    return s


def build_engine(model, gpu_util, tensor_parallel_size, max_num_seqs=8):
    from nanovllm import LLM
    return LLM(model, enforce_eager=True, tensor_parallel_size=tensor_parallel_size,
               attention_backend="triton_paged", max_model_len=2048,
               max_num_seqs=max_num_seqs, gpu_memory_utilization=gpu_util)


def live_seqs(llm):
    return list(llm.scheduler.running) + list(llm.scheduler.waiting)


def sampling_params(msg):
    from nanovllm import SamplingParams
    return SamplingParams(temperature=0, max_tokens=msg["max_tokens"],
                          ignore_eos=True)


# ---------------------------------------------------------------- collocated

def role_collocated(args):
    """Arm A: TP2 across both cards serving the framed client protocol."""
    llm = build_engine(args.model, args.gpu_util, tensor_parallel_size=2)
    srv = listen(args.port)
    print(f"[collocated] listening on {args.port}", flush=True)
    client, _ = srv.accept()
    client.setblocking(False)
    client_js = JsonStream()
    sel = selectors.DefaultSelector()
    sel.register(client, selectors.EVENT_READ)
    req_map = {}
    running = True
    while running:
        for key, _ in sel.select(timeout=0.05):
            try:
                chunk = key.fileobj.recv(1 << 20)
            except BlockingIOError:
                continue
            if not chunk:
                running = False
                break
            for msg in client_js.feed(chunk):
                if msg.get("type") == "done":
                    running = False
                    break
                seq_id = llm.add_request(msg["prompt"], sampling_params(msg))
                req_map[seq_id] = msg["req_id"]
        if not running:
            break
        if not live_seqs(llm):
            continue
        outputs, _ = llm.step()
        for seq_id, toks, metrics in outputs:
            send_json(client, {"type": "result", "req_id": req_map.pop(seq_id),
                               "tokens": list(toks), "metrics": metrics})
    client.close()
    llm.exit()
    print("[collocated] clean exit", flush=True)


# ---------------------------------------------------------------- decode (D)

def role_decode(args):
    import torch
    from nanovllm.engine.sequence import Sequence, SequenceStatus
    from nanovllm.pd.kv_channel import TcpKVChannel, KVFrameAssembler, import_kv

    llm = build_engine(args.model, args.gpu_util, tensor_parallel_size=1)
    kv_srv, ctrl_srv = listen(args.port), listen(args.port + 1)
    print(f"[decode] listening on {args.port}/{args.port + 1}", flush=True)
    kv_sock, _ = kv_srv.accept()
    ctrl_sock, _ = ctrl_srv.accept()
    for s in (kv_sock, ctrl_sock):
        s.setblocking(False)
    kvchan = TcpKVChannel(kv_sock, llm.model_runner.kv_cache, max_blocks=16)
    assembler = KVFrameAssembler()
    ctrl_js = JsonStream()
    bm = llm.scheduler.block_manager
    pending = deque()      # (req_id, seq) waiting for their KV frame
    fin_map = {}           # seq_id -> req_id
    sel = selectors.DefaultSelector()
    sel.register(kv_sock, selectors.EVENT_READ)
    sel.register(ctrl_sock, selectors.EVENT_READ)
    running = True
    while running:
        for key, _ in sel.select(timeout=0.05):
            try:
                chunk = key.fileobj.recv(1 << 22)
            except BlockingIOError:
                continue
            if not chunk:
                running = False
                break
            if key.fileobj is ctrl_sock:
                for msg in ctrl_js.feed(chunk):
                    if msg.get("type") == "done":
                        running = False
                        break
                    toks = list(msg["token_ids"])
                    seq = Sequence(toks, sampling_params(msg))
                    seq.num_prompt_tokens = seq.num_tokens - 1  # C1 = completion #1
                    seq.status = SequenceStatus.RUNNING
                    seq.is_prefill = False
                    seq.num_cached_tokens = seq.num_prompt_tokens
                    seq.block_table = [bm._allocate_block()
                                       for _ in range(msg["nblocks"])]
                    pending.append((msg["req_id"], seq))
                    print(f"[decode] admitted req{msg['req_id']} ctx={seq.num_tokens}",
                          flush=True)
            else:
                for (n, k_bytes, v_bytes) in assembler.feed(chunk):
                    req_id, seq = pending.popleft()
                    shape = (n,) + kvchan.buf.k.shape[1:]
                    arr_k = torch.frombuffer(bytearray(k_bytes), dtype=torch.int16)
                    arr_v = torch.frombuffer(bytearray(v_bytes), dtype=torch.int16)
                    kvchan.buf.k[:n] = arr_k.view(kvchan.buf.k.dtype).reshape(shape)
                    kvchan.buf.v[:n] = arr_v.view(kvchan.buf.v.dtype).reshape(shape)
                    import_kv(llm.model_runner.kv_cache, seq.block_table, kvchan.buf)
                    torch.musa.synchronize()
                    llm.scheduler.running.append(seq)
                    fin_map[seq.seq_id] = req_id
                    print(f"[decode] imported req{req_id} ({n} blocks)", flush=True)
        if not running:
            break
        if live_seqs(llm):
            outputs, _ = llm.step()
            for seq_id, toks, metrics in outputs:
                send_json(ctrl_sock, {"type": "result",
                                      "req_id": fin_map.pop(seq_id),
                                      "tokens": list(toks), "metrics": metrics})
    kv_sock.close()
    ctrl_sock.close()
    llm.exit()
    print("[decode] clean exit", flush=True)


# ---------------------------------------------------------------- prefill (P)

def role_prefill(args):
    from nanovllm.pd.kv_channel import TcpKVChannel

    llm = build_engine(args.model, args.gpu_util, tensor_parallel_size=1)
    kv_sock = connect_retry(args.host, args.d_port)
    ctrl_sock = connect_retry(args.host, args.d_port + 1)
    kvchan = TcpKVChannel(kv_sock, llm.model_runner.kv_cache, max_blocks=16)
    ctrl_js = JsonStream()
    srv = listen(args.port)
    print(f"[prefill] client on {args.port}, D at {args.d_port}", flush=True)
    client, _ = srv.accept()
    client.setblocking(False)
    client_js = JsonStream()
    t_add = {}
    req_seq = {}
    handed = set()
    prefill_s = {}
    sel = selectors.DefaultSelector()
    sel.register(client, selectors.EVENT_READ)
    sel.register(ctrl_sock, selectors.EVENT_READ)
    running = True
    while running:
        for key, _ in sel.select(timeout=0.05):
            try:
                chunk = key.fileobj.recv(1 << 20)
            except BlockingIOError:
                continue
            if not chunk:
                running = False
                break
            if key.fileobj is client:
                for msg in client_js.feed(chunk):
                    if msg.get("type") == "done":
                        running = False
                        break
                    t0 = time.perf_counter()
                    seq_id = llm.add_request(msg["prompt"], sampling_params(msg))
                    t_add[seq_id] = t0
                    req_seq[seq_id] = msg["req_id"]
            else:
                for msg in ctrl_js.feed(chunk):
                    if msg.get("type") == "result":
                        out = dict(msg)
                        out["prefill_s"] = prefill_s.get(msg["req_id"])
                        send_json(client, out)
        if not running:
            break
        if live_seqs(llm):
            llm.step()
        # hand off every sequence whose prefill just completed (C1 sampled)
        for seq in list(live_seqs(llm)):
            if seq.is_finished or seq.seq_id in handed or seq.seq_id not in req_seq:
                continue
            if seq.num_tokens <= seq.num_prompt_tokens:
                continue
            rid = req_seq[seq.seq_id]
            prefill_s[rid] = time.perf_counter() - t_add[seq.seq_id]
            send_json(ctrl_sock, {"type": "handoff", "req_id": rid,
                                  "token_ids": list(seq.token_ids),
                                  "max_tokens": seq.max_tokens,
                                  "nblocks": len(seq.block_table)})
            kvchan.send_blocks(seq.block_table)
            handed.add(seq.seq_id)
            llm.cancel_request(seq.seq_id)
            print(f"[prefill] handed off req{rid} prefill={prefill_s[rid]:.3f}s",
                  flush=True)
    client.close()
    llm.exit()
    print("[prefill] clean exit", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True,
                        choices=["collocated", "prefill", "decode"])
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-0.6B")
    parser.add_argument("--gpu-util", type=float, default=0.5)
    parser.add_argument("--port", type=int, default=29700,
                        help="collocated/prefill client port; decode kv port")
    parser.add_argument("--d-port", type=int, default=29720,
                        help="decode role's kv port (prefill connects here)")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    if args.role == "collocated":
        role_collocated(args)
    elif args.role == "decode":
        role_decode(args)
    else:
        role_prefill(args)


if __name__ == "__main__":
    main()
