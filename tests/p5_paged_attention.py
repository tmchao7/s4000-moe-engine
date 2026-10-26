"""MUSA correctness and poison-tail test for Triton paged decode attention."""

import math
import torch

from nanovllm.kernels.paged_attention import paged_decode_attention


def reference(q, k_cache, v_cache, block_tables, context_lens, scale):
    outputs = []
    q_heads = q.size(1)
    kv_heads = k_cache.size(2)
    group = q_heads // kv_heads
    for batch in range(q.size(0)):
        length = int(context_lens[batch].item())
        num_blocks = math.ceil(length / k_cache.size(1))
        pages = block_tables[batch, :num_blocks].long()
        keys = k_cache[pages].reshape(-1, kv_heads, q.size(2))[:length]
        values = v_cache[pages].reshape(-1, kv_heads, q.size(2))[:length]
        head_outputs = []
        for head in range(q_heads):
            kv_head = head // group
            scores = (keys[:, kv_head].float() @ q[batch, head].float()) * scale
            probs = torch.softmax(scores, dim=0)
            head_outputs.append((probs[:, None] * values[:, kv_head].float()).sum(0))
        outputs.append(torch.stack(head_outputs))
    return torch.stack(outputs).to(q.dtype)


def main():
    torch.musa.set_device(0)
    torch.manual_seed(20261026)
    batch, q_heads, kv_heads, dim, page = 4, 16, 2, 128, 256
    lengths = torch.tensor([127, 256, 385, 512], dtype=torch.int32, device="musa")
    block_tables = torch.tensor(
        [[7, 0], [3, 0], [6, 1], [4, 2]], dtype=torch.int32, device="musa"
    )
    k_cache = torch.randn(8, page, kv_heads, dim, dtype=torch.bfloat16, device="musa")
    v_cache = torch.randn_like(k_cache)
    q = torch.randn(batch, q_heads, dim, dtype=torch.bfloat16, device="musa")

    # Poison unused tails. A correct paged kernel never reads them.
    for row, length in enumerate(lengths.cpu().tolist()):
        tail = length % page
        if tail:
            page_id = int(block_tables[row, length // page].item())
            k_cache[page_id, tail:] = float("nan")
            v_cache[page_id, tail:] = float("nan")

    actual = paged_decode_attention(
        q, k_cache, v_cache, block_tables, lengths, dim ** -0.5, 512
    )
    expected = reference(q, k_cache, v_cache, block_tables, lengths, dim ** -0.5)
    torch.musa.synchronize()
    assert torch.isfinite(actual).all(), "poisoned KV tail leaked into output"
    max_abs = (actual.float() - expected.float()).abs().max().item()
    assert max_abs <= 2e-3, max_abs
    print(f"[p5-paged-attention] max_abs={max_abs:.6g} poison-tail PASS")


if __name__ == "__main__":
    main()
