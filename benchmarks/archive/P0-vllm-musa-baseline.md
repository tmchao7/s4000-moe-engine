# P0 基线：vllm-musa（预装 0.29.1）在 2×MTT S4000 上

- **日期**：2026-10-25
- **硬件**：2× MTT S4000（48GB，PCIe Gen5 x16，无 NVLink），驱动 5.2.0，MUSA SDK 5.2.0
- **软件**：vllm 0.29.1.dev0+g98dff2a.musa（实例预装）、torch 2.11.0.post1 + torch_musa、Python 3.10.12
- **模型**：Qwen/Qwen3-8B（dense，bf16，TP=2），`enforce_eager=True`，`gpu_memory_utilization=0.85`，`max_model_len=4096`
- **负载**：输入随机 100–1024 tokens，输出上限随机 100–1024 tokens，temperature=0.6（种子 42），脚本 `scripts/p0_vllm_baseline.py`

## 结果

| 并发 | 总输出 tokens | 耗时 (s) | 聚合吞吐 (tok/s) |
|---|---|---|---|
| 1   | 78     | 12.22  | 6.4 |
| 8   | 2,743  | 359.14 | 7.6 |
| 32  | 13,474 | 1604.19| 8.4 |
| 128 | （截断跳过） | — | — |

c=128 按 c=32 的速率推算需 >1h 且预期不改变结论（吞吐平台期 ~8–9 tok/s），故截断。

## 关键发现

1. **吞吐不随并发扩展**：1→32 并发聚合吞吐仅 6.4→8.4 tok/s。continuous batching 在该栈上未带来有效扩展——每序列 decode 速率被摊薄，说明瓶颈不在算力而在每步固定开销。
2. **每步耗时 ~120ms**：由 c=32 反推（8.4 steps/s 聚合）。粗算：TP2 每层 2 次 allreduce × 36 层 = 72 次/步，PCIe 上 MCCL allreduce 延迟 ~1ms 量级即 ~70ms/步，加上 bf16 权重读取（每卡 ~8GB → ~10ms@768GB/s）。**通信延迟占比疑似过半**——这正是自研引擎在 P2/P3 可优化的空间（通信融合、overlap、减少 allreduce 次数）。
3. **单流 decode 6.4 tok/s**：每卡有效带宽利用 ~7%（51GB/s / 768GB/s），attention/sampler 内核未做 decode 优化（enforce_eager，无 fused kernel）。
4. GPU 利用率 93%、温度 ~60C：GPU"忙"但效率极低，与上述分析一致。

## 对项目的意义

- 这套数字就是自研引擎 P1/P2 的**对照下限**：同等负载下打到 10–20 tok/s 聚合即显著超越官方栈。
- "官方 vllm-musa 在国产卡上吞吐不随并发扩展"本身是项目动机的核心证据，面试叙事直接可用。
- 待办（P1 复核）：对比 `enforce_eager=False`（graph capture）与 TP=1 单卡数字，定位瓶颈在通信还是内核。

## 环境备忘（P0 期间实测）

- 学术加速（`/etc/network_turbo`）在本地区不可用 → 模型下载走 `HF_ENDPOINT=https://hf-mirror.com` + `HF_HUB_DISABLE_XET=1`（xet 协议经镜像会 401）
- flashinfer 包 ABI 损坏（undefined symbol），不可用
- Triton（backend `mtgpu`）自定义 kernel 可运行 ✓
- MCCL 双卡 allreduce ✓；bf16 SDPA ✓
