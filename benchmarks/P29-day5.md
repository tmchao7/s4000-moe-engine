# P29 Day5：0.6B 双副本（DP=2）vs 单进程 TP2 原型

> 日期：2026-11-09。范围：P29-F 的架构原型 + 30B 单卡内存模型；不改生产默认。
> 原始 JSON：`benchmarks/data/P29/dp2_tp2*.json`、`dp2_dp2*.json`（`benchmarks/data/` 不入库）。
> 脚本：`scripts/p29_dp2_proto.py`。

## 1. 实验设计

- 模型用 **Qwen3-0.6B**（dense，不是 30B MoE），目的是验证架构/进程隔离，不把结果当 30B 性能。
- `tp2`：一个进程，`MUSA_VISIBLE_DEVICES=0,1`，`tensor_parallel_size=2`。
- `dp2`：两个独立进程，分别 `MUSA_VISIBLE_DEVICES=0/1`、`tensor_parallel_size=1`，各处理一半请求。
- 相同总负载：16 条 prompt、每条 128 prompt tokens、64 output tokens；`temperature=0`，`ignore_eos=True`。
- 每个 mode 跑 3 次（首次 + 2 次重复），比较 aggregate tok/s；每个 replica 都先 warmup 并清 prefix cache。

## 2. 结果：0.6B 上 DP2 约 1.20× TP2

| mode | run1 tok/s | run2 tok/s | run3 tok/s | median | CV |
|---|---:|---:|---:|---:|---:|
| TP2（单进程用双卡） | 237.48 | 232.75 | 228.56 | **232.75** | 1.6% |
| DP2（两个单卡副本） | 291.96 | 279.13 | 274.21 | **279.13** | 2.7% |

**DP2 / TP2 = 1.20×**（同总请求数/同总输出 token 数）。

细节：

- DP2 每个 replica 处理 8 条请求 × 64 token，耗时约 3.45–3.51 s；聚合 279 tok/s。
- TP2 一个进程处理 16 条 × 64 token，耗时约 4.31–4.48 s；聚合 233 tok/s。
- 两种 mode 的首 token 输出一致（脚本记录前 2 条的前 4 token）；完整 token 门未做，因为这是架构探针，不是 30B 产品切换。
- 进程隔离可用：`MUSA_VISIBLE_DEVICES=0/1` 在子进程里生效，两个副本能同时占两张卡；结束后双卡 0 MiB。

**不能直接外推 30B**：0.6B 是 dense、权重大小/通信占比/单卡算力占用都和 30B MoE 不同；1.20× 只说明「两个独立副本可以避免 TP2 的跨卡同步开销」，不代表 30B 也能 1.2×。

## 3. 30B DP=2 的内存账（纸面）

Day2 已测：checkpoint 29.03 GiB（FP8 27.84 + bf16 1.19），完整模型 FP8/W8 约 29GiB；bf16 反量化 56.88GiB。

| 项 | 当前 TP2 | DP=2 单副本（FP8/W8 保留） |
|---|---:|---:|
| 每卡权重 | 约 29.5 GiB packed | 约 29.0 GiB |
| KV block（256 token） | 12 MiB（2 KV 头/rank） | 24 MiB（4 KV 头全模型） |
| 当前 KV 配额 | 6.07 GiB/rank（518 blocks） | 假设 16 GiB 可用 → 约 680 blocks（约 174k tokens） |
| 聚合 KV 容量 | 2×518×256 ≈ **265k tokens** | 2×680×256 ≈ **348k tokens**（粗略） |
| 4096 context 可容纳 | 约 64 条 | 约 84 条（聚合） |

结论：

- **内存上 DP=2 可行**：FP8/W8 完整模型 29GiB + ~16GiB KV 在 48GB 内。
- **但引擎还没实现 FP8 权重保留**：当前 loader 在加载期反量化成 bf16，单卡 56.9GiB 放不下；所以 30B DP2 不是今天能直接跑的。
- **吞吐上不能直接复制 0.6B 的 1.20×**：DP2 每个副本要算完整 30B 的 active experts（3.3B），而 TP2 每卡只算一半；需要真实 30B FP8 保留原型才能判断。

## 4. 决策

1. **DP2 是当前唯一还没被证伪的架构方向**：0.6B 原型 1.20×，进程/设备隔离验证通过。
2. **30B DP2 的前置条件是 FP8 权重保留**（或 W8），不是继续调 TP/EP。
3. **不把 DP2 写进生产默认**；也不在 30B 上直接改 EP/AllToAll，因为那会改变数值路径。
4. 0.6B 结果只作为架构证据；30B 的 DP2 需要一次「FP8 保留 + 单卡完整模型」原型，属于更大的改动。

## 5. Day6 计划

- 先做 **FP8 保留的接口审计**：loader 是否能把 fp8 + scale_inv 留在设备上、MoE/attention 的 GEMM 接口需要哪些 dequant 入口；不改默认，只出可行性清单。
- 如果 Triton 内 dequant 不可用（Day2 已证）、MATE 也不可用，就明确 30B DP2 需要 C++/厂商支持，成本高于收益时停止 DP2。
- 备选：把 0.6B DP2 做成可复用的多副本部署脚本/文档，作为生产运维能力，而不是 30B 默认路径。