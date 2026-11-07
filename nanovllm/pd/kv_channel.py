"""Block-granular KV export/import for PD disaggregation (P15-C M1).

The engine's KV pool is ``kv_cache[2, L, blocks+1, block_size, kv_heads,
head_dim]`` (index 0 = K, 1 = V).  A request's KV is identified by its
``block_table`` (paged block ids).  Export copies whole blocks into a pinned
CPU staging buffer; import scatters the buffer back into a (possibly
different) block table on the peer side.  Whole-block granularity keeps both
directions as contiguous strided copies — no per-token gather — and matches
how the decode side allocates pages before import.

Bitwise contract: export→import is a pure copy; a round-trip must reproduce
the source blocks exactly (verified by tests/p15_pd_m1_kv.py).
"""
from __future__ import annotations

import struct

import torch


class KVBlockBuffer:
    """Pinned CPU staging for N whole KV blocks.

    Layout: k/v tensors of shape [n_blocks, L, block_size, kv_heads, head_dim]
    — per-block slices of the pool with the block axis pulled to the front.
    """

    def __init__(self, n_blocks: int, n_layers: int, block_size: int,
                 kv_heads: int, head_dim: int, dtype=torch.bfloat16):
        shape = (n_blocks, n_layers, block_size, kv_heads, head_dim)
        self.k = torch.empty(shape, dtype=dtype, pin_memory=True)
        self.v = torch.empty(shape, dtype=dtype, pin_memory=True)
        self.n_blocks = n_blocks

    @classmethod
    def from_pool(cls, kv_cache: torch.Tensor, n_blocks: int) -> "KVBlockBuffer":
        _, n_layers, _, block_size, kv_heads, head_dim = kv_cache.shape
        return cls(n_blocks, n_layers, block_size, kv_heads, head_dim,
                   dtype=kv_cache.dtype)

    @property
    def nbytes(self) -> int:
        return (self.k.element_size() * self.k.numel()
                + self.v.element_size() * self.v.numel())


def extract_kv(kv_cache: torch.Tensor, block_table: list[int],
               buf: KVBlockBuffer) -> None:
    """Copy the blocks named by ``block_table`` (device) into ``buf`` (pinned).

    I28 SOP: a cross-block strided view (``pool[:, bid]``) copied D2H through
    pinned staging intermittently returns garbage on this MUSA stack — gather
    with ``index_select`` (contiguous output) first, then one pinned copy.
    """
    assert len(block_table) <= buf.n_blocks
    n = len(block_table)
    k_pool, v_pool = kv_cache[0], kv_cache[1]          # [L, blocks+1, bs, kvh, d]
    idx = torch.tensor(block_table, device=k_pool.device, dtype=torch.long)
    kg = k_pool.index_select(1, idx)                   # [L, n, bs, kvh, d]
    vg = v_pool.index_select(1, idx)
    buf.k[:n].copy_(kg.permute(1, 0, 2, 3, 4))
    buf.v[:n].copy_(vg.permute(1, 0, 2, 3, 4))


def import_kv(kv_cache: torch.Tensor, block_table: list[int],
              buf: KVBlockBuffer) -> None:
    """Scatter ``buf`` blocks into the blocks named by ``block_table`` (I28-safe)."""
    assert len(block_table) <= buf.n_blocks
    n = len(block_table)
    k_pool, v_pool = kv_cache[0], kv_cache[1]
    idx = torch.tensor(block_table, device=k_pool.device, dtype=torch.long)
    kdev = buf.k[:n].to(k_pool.device).permute(1, 0, 2, 3, 4).contiguous()
    vdev = buf.v[:n].to(k_pool.device).permute(1, 0, 2, 3, 4).contiguous()
    k_pool.index_copy_(1, idx, kdev)
    v_pool.index_copy_(1, idx, vdev)


class LocalKVChannel:
    """M1 channel: device → pinned → device within one process/one card.

    Stages through pinned host memory exactly like a cross-process channel
    would, so M1 measures the copy path a real PD deployment pays — only the
    socket leg is missing.
    """

    def __init__(self, kv_cache: torch.Tensor):
        self.kv_cache = kv_cache
        self.bytes_moved = 0

    def send(self, block_table: list[int]) -> KVBlockBuffer:
        buf = KVBlockBuffer.from_pool(self.kv_cache, len(block_table))
        extract_kv(self.kv_cache, block_table, buf)
        torch.musa.synchronize()
        self.bytes_moved += buf.nbytes
        return buf

    def recv(self, block_table: list[int], buf: KVBlockBuffer) -> None:
        import_kv(self.kv_cache, block_table, buf)
        torch.musa.synchronize()
        self.bytes_moved += buf.nbytes


def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    """bf16-safe tensor → bytes (numpy lacks bf16; view as int16, lossless)."""
    return t.contiguous().view(torch.int16).numpy().tobytes()


def _bytes_to_tensor(data: bytes, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    arr = torch.frombuffer(bytearray(data), dtype=torch.int16)
    return arr.view(dtype).reshape(shape)


class TcpKVChannel:
    """M2 channel: block-granular KV over TCP with pinned CPU staging.

    One socket, length-prefixed frames: [n_blocks:u32][k_bytes:u32][v_bytes:u32]
    then raw K then raw V.  The pinned staging buffer is reused across sends,
    so steady-state cost is two D2H/H2D copies + one socket hop per block
    (~29.4 MiB/block for the 0.6B production layout, 96 KiB/token).
    """

    MAGIC_HEADER = struct.Struct("<III")

    def __init__(self, sock, kv_cache: torch.Tensor, max_blocks: int):
        self.sock = sock
        self.kv_cache = kv_cache
        self.buf = KVBlockBuffer.from_pool(kv_cache, max_blocks)
        self.bytes_moved = 0

    def send_blocks(self, block_table: list[int]) -> None:
        n = len(block_table)
        assert n <= self.buf.n_blocks
        extract_kv(self.kv_cache, block_table, self.buf)
        torch.musa.synchronize()
        k_bytes = _tensor_to_bytes(self.buf.k[:n])
        v_bytes = _tensor_to_bytes(self.buf.v[:n])
        self.sock.sendall(self.MAGIC_HEADER.pack(n, len(k_bytes), len(v_bytes)))
        self.sock.sendall(k_bytes)
        self.sock.sendall(v_bytes)
        self.bytes_moved += len(k_bytes) + len(v_bytes)

    def recv_blocks(self, block_table: list[int]) -> None:
        n = len(block_table)
        header = self._recv_exact(self.MAGIC_HEADER.size)
        n_decl, k_len, v_len = self.MAGIC_HEADER.unpack(header)
        assert n_decl == n, f"block count mismatch {n_decl} != {n}"
        k_data = self._recv_exact(k_len)
        v_data = self._recv_exact(v_len)
        shape = (n,) + self.buf.k.shape[1:]
        self.buf.k[:n] = _bytes_to_tensor(k_data, shape, self.buf.k.dtype)
        self.buf.v[:n] = _bytes_to_tensor(v_data, shape, self.buf.v.dtype)
        import_kv(self.kv_cache, block_table, self.buf)
        torch.musa.synchronize()
        self.bytes_moved += k_len + v_len

    def _recv_exact(self, n: int) -> bytes:
        chunks = []
        while n > 0:
            chunk = self.sock.recv(min(n, 1 << 22))
            if not chunk:
                raise ConnectionError("peer closed during KV transfer")
            chunks.append(chunk)
            n -= len(chunk)
        return b"".join(chunks)
