# P29 Day6：FP8 权重保留的接口审计与成本测量

> 日期：2026-11-09。范围：P29-G 的 FP8 保留可行性；不改生产默认。
> 原始 JSON：`benchmarks/data/P29/fp8_dequant_cost.json`（`benchmarks/data/` 不入库）。
> 脚本：`scripts/p29_fp8_dequant_cost.py`。

## 1. 为什么要做这个审计

Day5 的 30B DP=2 内存账成立的前提是「完整模型以 FP8/W8 留在单卡」。当前 loader 不是这样：

- `nanovllm/utils/loader.py:28` `dequant_fp8_block()` 在加载期把 FP8 权重乘 `scale_inv` 变成 bf16；
- `:106` 每个 `*_scale_inv` 被消费，`:112` 写入 bf16 参数；
- 结果：checkpoint 29.03GiB → 模型 bf16 权重约 56.9GiB；TP2 每 rank 约 29.5GiB，单卡放不下完整模型。

要保留 FP8，需要同时改：

| 层 | 当前 | FP8 保留需要 | 阻塞点 |
|---|---|---|---|
| loader | CPU 反量化成 bf16 | 保留 fp8 + scale_inv 到设备；按 TP/EP 分片 | 参数存储、packed expert 布局要改 |
| attention 线性层 | bf16 GEMM | dequant+GEMM | 需要自定义 kernel |
| MoE prefill | `torch._grouped_mm` bf16 | FP8 grouped GEMM | 厂商 op 不支持 FP8 |
| MoE decode | `bmm_dense` bf16 | FP8 bmm / dequant+bmm | 同上 |
| KV | bf16；int8 实验失败 | 不变 | — |

## 2. 成本实测：每层 dequant 就超过整个 decode step

在 MUSA 上用真实 30B 的**本地专家层**形状做合成 dequant（FP8 e4m3 + 128×128 scale_inv → bf16）：

| 张量 | 形状 | FP8 大小 | dequant 时间 | 峰值显存 |
|---|---:|---:|---:|---:|
| gate_up（64 专家/rank） | [64,2048,1536] | 192 MiB | **10.15 ms** | 2880 MiB |
| down（64 专家/rank） | [64,768,2048] | 96 MiB | **5.23 ms** | 1440 MiB |
| 每层合计 | — | 288 MiB | **15.38 ms** | — |
| 48 层 × 每 forward | — | 13.5 GiB | **738.0 ms** | — |

当前 30B decode 步时约 **142 ms**；**仅 dequant 就要 738 ms/forward**，还没算 GEMM/attention/all-reduce。

> 注：过程里出现 `muDNN ... data type is not supported` 的告警（FP8 cast 走 fallback），时间仍可测；这本身也说明当前栈对 FP8 cast 不是一等公民。

## 3. 三条路都不通

| 路径 | 结论 |
|---|---|
| 保留 FP8 + 每 forward dequant | 738ms/forward，**不可行** |
| 保留 FP8 + Triton 内 fused dequant | Day2 已证 Triton-mtgpu 报 `__mt_tt_e4m3_to_f32` relocation，**编译失败** |
| 保留 FP8 + MATE 8-bit | Day3 已确认 MATE 未安装、无库；**当前不可用** |

因此，30B DP=2 在当前 MUSA 栈上的前置条件（FP8 保留 + 高效 dequant GEMM）**不成立**；要做只能写 MUSA C++/汇编或拿到厂商 MATE，成本远超 9 天窗口。

## 4. 决策

1. **30B DP=2：NO-GO**（当前栈）。不是内存不够，而是没有可用的 FP8 保留 + fused dequant 路径。
2. **量化线收口**：保留 Day1–Day3 的结论（显存模型、ATen INT8 慢、W8 原型慢、MATE 缺失），不再做 4-bit e2e。
3. **生产保持 TP2+EP2 + bf16 反量化**；0.6B DP2 的 1.20× 只作为架构证据/运维参考，不用于 30B。
4. 剩余天数如果继续，应该做 **不改数值路径的系统能力**（异步调度/文档/运维）或收口；不再追 30B DP2/4-bit。

## 5. Day7 候选（如果继续）

- **异步调度/overlap**：评估现有 overlap 调度的 opt-in 收益，不碰数值路径；
- **收口包**：把 Day1–Day6 的结论整理成面试/交接材料，明确「做了什么、否了什么、为什么」。
- 如果没有新的产品需求，建议在 Day6 后停止实验，避免为了填满 9 天而做低 ROI 的 kernel。