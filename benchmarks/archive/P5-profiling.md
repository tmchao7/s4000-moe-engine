# P5 — 指标校准与真实 30B profiling

> 日期：2026-10-26  
> 硬件：2× MTT S4000 48GB，单机 PCIe  
> 模型：Qwen3-30B-A3B-Instruct-2507-FP8，加载期反量化为 BF16，TP2+EP2  
> 原始证据：`benchmarks/data/p5_profile.json`（不入 Git），SHA256 `147e7761799863017210d74f5da236cc07f027971c4eb7afc4f95fe7bdbc089d`  
> 可审计决策摘要：`benchmarks/p5-profile-decision.json`

## 1. 指标口径校准

P2/P3/P4 的原始 JSON 已冻结，不回写历史证据。P5 新增 schema v2，并明确以下差异：

- P2 报告中旧名“p95 TPOT”实际是把所有 token gap 打平后的 p95，P5 起统一称为 **p95 ITL**。
- 请求级 TPOT 按每个请求 `(E2E - TTFT) / (output_tokens - 1)` 计算，再对请求取分位数；输出少于两个 token 时为 `null`。
- CV 使用 `sample standard deviation / arithmetic mean`；P2 legacy v1 使用的是 `pstdev / median`，两者不能混比。
- 在线成功请求必须收到 SSE `[DONE]`，token 数来自服务端 usage 或相同 tokenizer 的本地计数，不能用非空 SSE chunk 数代替。
- schema v2 同时报 request/input/output/total throughput、TTFT、TPOT、ITL、E2E、最大并发和 SLO goodput。
- Prometheus 使用 counter/gauge/histogram，并带 `HELP/TYPE`；不在进程内固化不可聚合的服务端分位数。

因此，P2 的 grouped-MM 3.0–3.2× 加速和 74.0 tok/s @b32 仍是可信历史吞吐结果，但其 legacy 延迟和 CV 字段只按当时口径解释。P3/P4 的旧在线数据可证明 mixed scheduling 机制生效，不作为工业级容量结论。

## 2. profiling 方法与正确性门

`scripts/p5_profile.py` 在一次模型加载中执行：

1. 用仓库内 `benchmarks/p2-reference-8tok.json` 对 loop/grouped-MM 各运行两次；逐 token 全等后才继续。
2. 覆盖 decode batch 1/8/16/32、context 128/512/2048（长上下文为 batch 16/32）。
3. 两个 rank 分别以 MUSA Event 记录 attention、MoE 和 EP collective；一个 snapshot 只统一 synchronize 一次。
4. 按预先写死的门槛自动选路线，避免看完结果再改判断标准。

本次输出：`[p5-profile] 30B correctness gate PASS`。profiling 插桩会增加 event/记录开销，以下时间只用于占比和路线选择，不能当作最终吞吐。

## 3. 设备侧结果

| batch | context | attention / forward | MoE local / forward | EP all-reduce / forward | rank pair 不平衡 |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 7.62% | 49.14% | 5.45% | 1.69% |
| 8 | 128 | 9.15% | 50.33% | 2.98% | 2.27% |
| 16 | 128 | 10.94% | 53.31% | 3.67% | 3.11% |
| 32 | 128 | 15.13% | 53.48% | 3.56% | 4.28% |
| 16 | 512 | 20.96% | 46.10% | 2.97% | 2.89% |
| 32 | 512 | 31.53% | 40.88% | 3.23% | 4.39% |
| 16 | 2048 | 45.77% | 31.25% | 2.48% | 4.43% |
| 32 | 2048 | **58.56%** | 27.51% | 4.97% | 5.19% |

最关键的 b32/context 2048，8 个 decode step、48 层累计的主要片段为：

| 片段 | 时间 |
|---|---:|
| forward | 7915.5 ms |
| paged KV gather + mask/expand | **3571.8 ms** |
| SDPA | 1063.5 ms |
| grouped GEMM 1 | 700.4 ms |
| grouped GEMM 2 | 562.0 ms |
| route/pack | 558.8 ms |
| deterministic combine | 22.6 ms |
| EP all-reduce（较慢 rank） | 393.0 ms |

## 4. 自动决策

预设门槛与实测结论：

| 路线 | 门槛 | 最大实测 | 结论 |
|---|---:|---:|---|
| paged attention | attention ≥30% forward | **58.56%** | 触发，第一优先级 |
| MoE fusion | MoE ≥35% forward 或 route/combine ≥10% MoE | 53.48% / 44.93% | 触发，第二优先级 |
| collective overlap | all-reduce ≥20% forward | 5.45% | 拒绝，不投入 |
| profiled placement | rank pair 不平衡 >10% | 5.19% | 拒绝，不投入 |

结论不是“偏好 Triton”，而是长上下文下 `gather + SDPA` 已占 forward 的 31.5%–58.6%，且随 context 明显增长。下一次 30B 加载必须先做 `sdpa` 与 `triton_paged` 的同进程正确性/性能 A/B。只有所有 token 完全一致、长上下文 b16/b32 请求级 p95 TPOT 至少改善 10%、吞吐不下降且 CV≤5%，才允许更改默认后端。

## 5. MoE 社区方案的边界

- MoonEP 的动态冗余专家与标准 dispatch/combine EP 面向更多 GPU；本项目是双卡 replicated hidden + local experts + all-reduce，不直接集成。
- UniEP 的确定性 token mapping、融合边界和 shape-driven autotune 可借鉴，但其 Hopper 集群训练实现不能直接用于 MUSA。
- DeepGEMM MegaMoE 面向 CUDA SM90/SM100 的 FP8/FP4；本项目只借鉴 `dispatch → GEMM → SwiGLU → combine` 的融合划分，不声称移植。
- 当前 `fused_grouped_mm` 只是候选后端：保留两次原生 `torch._grouped_mm`，融合 SwiGLU 与 deterministic weighted-combine；尚未融合 route/pack，不能称为 MegaMoE。

## 6. 下一实验

精确命令、加载预算和失败处理见 `docs/HANDOFF.md`。默认值目前仍是：

```text
attention_backend = sdpa
moe_backend = grouped_mm
expert_placement = linear
```

在真实 30B A/B 达标之前不得更改。
