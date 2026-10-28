"""MUSA correctness and poison-tail tests for Triton paged decode attention."""

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


def assert_matches_reference(q, k_cache, v_cache, block_tables, lengths, scale, label):
    actual = paged_decode_attention(q, k_cache, v_cache, block_tables, lengths, scale, int(lengths.max().item()))
    expected = reference(q, k_cache, v_cache, block_tables, lengths, scale)
    torch.musa.synchronize()
    assert torch.isfinite(actual).all(), f"{label}: poisoned KV tail leaked into output"
    max_abs = (actual.float() - expected.float()).abs().max().item()
    assert max_abs <= 2e-3, f"{label}: {max_abs}"
    return max_abs


def test_small_poison_tail():
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

    max_abs = assert_matches_reference(
        q, k_cache, v_cache, block_tables, lengths, dim ** -0.5, "small poison-tail"
    )
    return max_abs


def test_page_boundary_matrix():
    """Exercise every relevant side of page and 32-token kernel boundaries."""
    torch.manual_seed(20261027)
    lengths_cpu = [255, 256, 257, 511, 512, 513, 2047, 2048, 2049]
    batch, q_heads, kv_heads, dim, page = len(lengths_cpu), 16, 2, 128, 256
    blocks_per_row = [math.ceil(length / page) for length in lengths_cpu]
    total_pages = sum(blocks_per_row)
    block_tables = torch.zeros(batch, max(blocks_per_row), dtype=torch.int32, device="musa")
    next_page = 0
    for row, block_count in enumerate(blocks_per_row):
        # Each row owns unique, deliberately non-contiguous physical pages.
        pages = list(range(next_page, next_page + block_count))
        pages[::2] = list(reversed(pages[::2]))
        block_tables[row, :block_count] = torch.tensor(pages, dtype=torch.int32, device="musa")
        next_page += block_count
    assert next_page == total_pages

    k_cache = torch.randn(total_pages, page, kv_heads, dim, dtype=torch.bfloat16, device="musa")
    v_cache = torch.randn_like(k_cache)
    q = torch.randn(batch, q_heads, dim, dtype=torch.bfloat16, device="musa")
    lengths = torch.tensor(lengths_cpu, dtype=torch.int32, device="musa")
    for row, length in enumerate(lengths_cpu):
        tail = length % page
        if tail:
            page_id = int(block_tables[row, length // page].item())
            k_cache[page_id, tail:] = float("nan")
            v_cache[page_id, tail:] = float("nan")
    return assert_matches_reference(
        q, k_cache, v_cache, block_tables, lengths, dim ** -0.5, "page-boundary matrix"
    )


def main():
    small_max = test_small_poison_tail()
    boundary_max = test_page_boundary_matrix()
    print(
        f"[p5-paged-attention] small_max_abs={small_max:.6g} "
        f"boundary_max_abs={boundary_max:.6g} poison-tail PASS"
    )


if __name__ == "__main__":
    main()
