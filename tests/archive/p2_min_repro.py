"""Layer-math repro: TP1 vs emulated-TP2 MoE forward on the tiny checkpoint.

Runs two real processes (gloo, world_size=2) over the actual engine code path
but with a tiny model, comparing per-layer activations against a TP1
reference computed in a separate single-rank process. This isolates the
layer math + real dist collectives from the engine's spawn/shm plumbing.

    PYTHONPATH=/root/moe-server-s4000 python3 tests/p2_min_repro.py
"""
import os
import sys

import torch
import torch.distributed as dist
from types import SimpleNamespace

sys.path.insert(0, "/root/moe-server-s4000")

MODEL_DIR = "/tmp/tiny_moe"


def build_model():
    from transformers import Qwen3MoeConfig
    from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM
    from nanovllm.utils.loader import load_model

    cfg = Qwen3MoeConfig.from_pretrained(MODEL_DIR)
    torch.set_default_dtype(cfg.torch_dtype)
    runtime = SimpleNamespace(
        expert_parallel_size=dist.get_world_size(),
        moe_backend="loop",
        expert_placement="linear",
        expert_map_path=None,
        collect_expert_stats=False,
    )
    model = Qwen3MoeForCausalLM(cfg, runtime)
    load_model(model, MODEL_DIR)
    return model, cfg


def forward_layer0_mlp(model, x):
    with torch.no_grad():
        return model.model.layers[0].mlp(x)


def forward_full(model, input_ids, positions):
    """Full model forward with in-loop (non-paged) attention."""
    from nanovllm.utils.context import set_context, reset_context
    T = input_ids.numel()
    cu = torch.arange(0, T + 1, dtype=torch.int32)
    for m in model.modules():
        if hasattr(m, "k_cache"):
            m.k_cache = torch.tensor([])
            m.v_cache = torch.tensor([])
    set_context(True, cu, cu, T, T, torch.full((T,), -1, dtype=torch.int32), None, None)
    h = model(input_ids, positions)
    reset_context()
    return h


def tp1_reference(x, input_ids, positions):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29701"
    dist.init_process_group("gloo", rank=0, world_size=1)
    from nanovllm.layers.rotary_embedding import get_rope
    get_rope.cache_clear()
    model, _ = build_model()
    with torch.no_grad():
        mlp_out = forward_layer0_mlp(model, x)
        h = forward_full(model, input_ids, positions)
        logits = model.compute_logits(h)
    dist.destroy_process_group()
    get_rope.cache_clear()
    return mlp_out, h, logits


def tp2_worker(rank, x, input_ids, positions):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29702"
    dist.init_process_group("gloo", rank=rank, world_size=2)
    from nanovllm.layers.rotary_embedding import get_rope
    get_rope.cache_clear()
    model, _ = build_model()
    with torch.no_grad():
        mlp_out = forward_layer0_mlp(model, x)
        h = forward_full(model, input_ids, positions)
        logits = model.compute_logits(h)
    dist.destroy_process_group()
    get_rope.cache_clear()
    return mlp_out, h, logits


def main():
    torch.manual_seed(0)
    x = torch.randn(5, 64)
    input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    positions = torch.arange(8)

    mlp_ref, h_ref, logits_ref = tp1_reference(x, input_ids, positions)

    # real 2-process TP over the same code path
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for r in range(2):
        p = ctx.Process(target=_worker_entry, args=(r, x, input_ids, positions, q))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    res = {}
    while not q.empty():
        rank, payload = q.get()
        res[rank] = [torch.from_numpy(a) if hasattr(a, "dtype") else a for a in payload]

    mlp0, h0, logits0 = res[0]
    mlp1, h1, logits1 = res[1]

    for name, ref, a, b in [
        ("layer0-mlp", mlp_ref, mlp0, mlp1),
        ("final-hidden", h_ref, h0, h1),
    ]:
        # each rank's output already contains the allreduced sum of both
        # partials, so rank0's tensor alone should equal the TP1 reference
        d = (a.float() - ref.float()).abs()
        rel = d.max() / ref.float().abs().max().clamp_min(1e-9)
        print(f"[repro] {name}: TP2-rank0 vs TP1 max_diff={d.max():.3e} rel={rel:.3e}")
        print(f"[repro]   ref[0,:5]:  {ref[0,:5].tolist()}")
        print(f"[repro]   r0[0,:5]:   {a[0,:5].tolist()}")

    # logits: rank0 gathers full vocab; compare directly
    if logits0 is not None:
        d = (logits0.float() - logits_ref.float()).abs()
        rel = d.max() / logits_ref.float().abs().max().clamp_min(1e-9)
        print(f"[repro] logits: TP2-rank0 vs TP1 max_diff={d.max():.3e} rel={rel:.3e}")
        print(f"[repro]   argmax ref: {logits_ref.float().argmax(-1).tolist()}")
        print(f"[repro]   argmax r0:  {logits0.float().argmax(-1).tolist()}")


def _worker_entry(rank, x, input_ids, positions, q):
    out = tp2_worker(rank, x, input_ids, positions)
    # numpy copies — torch shared-storage reduction fails after worker exit
    payload = [t.cpu().numpy().copy() if torch.is_tensor(t) else t for t in out]
    q.put((rank, payload))


if __name__ == "__main__":
    main()
