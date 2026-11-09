# P29 Day8：长上下文 / KV 容量探测（30B）

> 日期：2026-11-09。范围：P29-H 长上下文能力；不改生产默认。
> 原始 JSON：`benchmarks/data/P29/long_context.json`、`long_context_16k.json`（`benchmarks/data/` 不入库）。
> 脚本：`scripts/p29_long_context.py`。

## 1. 实验设计

- 30B、TP2/EP2、生产栈：`triton_paged` + `grouped_mm` prefill + `bmm_dense` decode + eager + `stall_free` + chunk 256；
- 随机 token prompt，单请求，先 prefill，再 decode 8 步；无参考 token，**这是容量/性能探测，不是质量门**；
- 两次加载：
  - 1k/4k/8k：`max_model_len=8208`；
  - 16k：`max_model_len=16400`（16k prompt + decode 余量）；
- `max_num_batched_tokens=16384`，`gpu_memory_utilization=0.85`。

## 2. 结果

| context | prefill 时间 | prefill tok/s | decode median | peak 显存 |
|---:|---:|---:|---:|---:|
| 1024 | 1.697 s | 603.4 | **158.6 ms** | 37.1 GiB |
| 4096 | 6.675 s | 613.7 | **173.5 ms** | 37.2 GiB |
| 8192 | 13.956 s | 587.0 | **210.8 ms** | 37.2 GiB |
| 16384 | 30.912 s | 530.0 | **284.8 ms** | 35.8 GiB |

KV block 日志：

- 1k/4k/8k 加载：**651 blocks**，block 12 MiB，KV 7.63 GiB/rank；
- 16k 加载：**525 blocks**，KV 6.15 GiB/rank；
- block 数会随当次 `used/peak` 与 `max_model_len` 的 buffer 一起浮动，不要跨加载直接比较。

容量换算（按 block=256 token）：

- 651 blocks ≈ 166k tokens：1k context 约 166 条、4k 约 41 条、8k 约 20 条；
- 525 blocks ≈ 134k tokens：16k context 约 8 条；
- 当前生产默认 `max_model_len=4096`，本身没有把 KV 用满；8k/16k 不会 OOM。

## 3. 解读

- **8k/16k 能跑**：prefill 530–614 tok/s，decode 从 1k 的 158.6ms 增至 16k 的 284.8ms（约 1.8×），没有崩溃/OOM。
- **16k 的 TTFT 很高**：单请求 prefill 30.9s，交互场景不可接受；适合离线/批处理，或需要更激进的 chunk/并行 prefill 优化。
- **8k 是更现实的产品候选**：prefill 14s、decode 211ms，仍明显高于 4k 的 173ms，但比 16k 可控。
- **质量未测**：随机 token 只能证明 kernel/显存不炸，不能证明长文推理质量；产品化前要加真实长文档 eval。

## 4. 决策

1. 不修改生产默认 `max_model_len=4096`；8k/16k 作为 **opt-in 能力**记录。
2. 如果要做 8k 产品：需要补真实长文质量门 + 输入长度校验 + 显存/SLO 压测；不能只凭本探测。
3. 16k 只建议离线/研究用途；交互 SLO 会被 31s prefill 破坏。
4. 不做 32k；除非有明确长文档需求并且愿意接受 prefill 成本。