# 架构设计

> 状态：P2 正确性 + 性能、P3 调度、P4 服务化均已实现并通过设备侧回归；数字见 `benchmarks/`。

## 1. 目标与约束

- **硬件**：2× MTT S4000（每卡 48GB、56 SM、768GB/s、PCIe Gen5 x16，无卡间 NVLink/MTLink 实测依赖，TP 通信走 PCIe）。
- **算力限制**：S4000 无 FP8 算力，bf16 为主要计算精度；Triton 存在官方 mtgpu backend（已实测可用）。
- **软件栈**：torch 2.11 + torch_musa 2.11、MCCL 2.11.4（NCCL 等价物，已实测双卡 allreduce）、Triton-mtgpu 3.2、transformers 5.13、Python 3.10。
- **存储限制**：数据盘仅 50GB → 采用官方 FP8 checkpoint（~31.2GB），加载时反量化为 bf16（显存充裕：96GB 总量）。
- **对照组**：实例预装 vllm-musa 0.29.1（含编译好的 MoE C 扩展），作为性能/正确性参照与动机素材（详见 `benchmarks/archive/P0-vllm-musa-baseline.md`）。

## 2. 模块设计（与代码一致）

```
nanovllm/
├── config.py                  # 运行时配置（TP/EP、MoE 后端、专家放置、调度策略、metrics）
├── llm.py                     # 同步入口（LLM = LLMEngine）
├── async_engine.py            # 后台线程引擎 + 命令队列（服务化用）
├── sampling_params.py
├── engine/
│   ├── llm_engine.py          # 主循环、请求生命周期、step 级指标与 JSONL trace
│   ├── scheduler.py           # FCFS / stall_free / stall_free_lpm；ScheduledBatch
│   ├── block_manager.py       # paged KV + block-hash 前缀缓存 + KV 计数器
│   ├── model_runner.py        # musa 设备抽象、shm 多进程、prefill/decode 批构造
│   └── sequence.py            # 序列状态 + 请求级 TTFT/TPOT 指标
├── entrypoints/openai/
│   └── api_server.py          # /health /v1/models /v1/chat/completions(SSE) /metrics
├── models/
│   ├── qwen3.py               # 上游 dense 实现
│   └── qwen3_moe.py           # ★ softmax top-8 router、packed experts、EP2、grouped-MM
├── layers/
│   ├── attention.py           # MUSA SDPA 的 paged prefill/decode（含 KV 尾部零初始化）
│   ├── linear.py              # Column/RowParallel（MCCL all-reduce）
│   ├── layernorm.py           # RMSNorm（out-of-place，避免污染残差流）
│   └── ...
└── utils/
    ├── compat.py              # MUSA 设备/流兼容层
    ├── context.py             # 批上下文（cu_seqlens/block_tables/slot_mapping）
    └── loader.py              # FP8(e4m3+block scale) → bf16 反量化加载 + 全覆盖断言
```

配套：`scripts/`（离线消融、专家画像、在线压测、渲染）、`tests/`（正确性/调度/服务回归）、`benchmarks/`（各阶段数字与结论）。

## 3. 关键技术决策

| 决策点 | 选择 | 理由 / 风险 |
|---|---|---|
| 基础引擎 | nano-vllm fork（保留上游 git 历史） | 1.4k 行可完全吃透；调度器简单，扩展空间大；与 mini-sglang（5k 行、调度已完备）对比后选定 |
| 模型 | Qwen3-30B-A3B-Instruct-2507-FP8 | 认知度最高；标准 softmax-top-k MoE；FP8 版本解决 50G 磁盘限制；反量化加载本身是引擎经典组件 |
| 注意力实现 | SDPA（bf16）起步 → Triton-mtgpu paged attention 打磨 | SDPA 已在 MUSA 实测可用；起步期正确性优先，性能差距留给内核阶段 |
| MoE 并行策略 | Attention TP2 + 专家 EP2；hidden replicated，专家输出 all-reduce | 两卡 PCIe 场景无需 All-to-All；每卡 64 个完整专家 |
| 专家计算 | packed `[expert,input,output]` 权重 + `torch._grouped_mm` | 单份连续存储（不额外复制权重），消除 64 次 Python 循环与逐专家同步；loop 后端保留作数值参考 |
| 专家放置 | `linear` / `round_robin` / `profiled` | profiled 由路由画像生成静态均衡映射，且带 `config.json` 哈希守卫，防止串用别的模型的地图 |
| 调度 | FCFS 基线 + stall-free（decode-first，bounded prefill chunk） | 与 Sarathi-Serve 的 stall-free batching 同思路：先保 decode 不被打断，再用剩余 token budget 填充 prefill |
| 前缀缓存 | block-hash 前缀缓存 + longest-prefix admission（`stall_free_lpm`） | 命中率可测量（`kv_prefix_hit_tokens_total`），冷/热可对比 |
| 量化加载 | FP8 e4m3 + block scale → bf16，加载期一次性反量化 | 纯 PyTorch 实现，无需自定义 kernel；S4000 无 FP8 算力故不保留 fp8 运行时 |
| CUDA graph | 目前 `enforce_eager` | torch_musa graph capture 支持度未知，非关键路径 |

## 4. 数据流（一次 generate 请求）

```
请求 → Scheduler（decode-first / 前缀缓存 / mixed chunked-prefill batch）
     → ModelRunner（musa 设备）
         → embed → [MoE layer × N]:
             attention(SDPA + paged KV) → allreduce
             router(softmax top-8) → grouped expert GEMM → EP allreduce
         → sampler → token 回填 sequence
     → 输出流
```

TP2 通信量：每层 2 次 allreduce（attention 后 + MoE 后）。PCIe Gen5 x16 下隐藏通信不是瓶颈；实测吞吐随 batch 线性扩展（见 `benchmarks/archive/P2-moe-tp2.md`）。

## 5. 正确性策略

分层设门，任何一层失败都不进入性能测量：

1. **逐层数值参考**：`tests/p2_ground_truth.py` 用手写 PyTorch（CPU、无 KV cache、无 TP）对 tiny checkpoint 做贪心生成；tiny 模型的 TP1 输出必须与它逐 token 一致。
2. **并行等价**：`tests/p2_tp_check.py`（TP1==TP2）、`tests/p2_grouped_mm.py`（TP1/TP2 × loop/grouped_mm，外加 round_robin 放置）必须四路 token 完全一致。
3. **解码确定性**：`tests/p2_determinism.py` 反复重建引擎、不做 KV 清零，要求多次结果相同（复现并锁死 MUSA SDPA 读 KV 尾部未初始化数据的 bug）。
4. **30B 端到端**：固定 4 条中英文 prompt 贪心输出连贯；`scripts/p2_benchmark.py --verify-prompts` 会把 token id 与冻结的参考逐 id 比对，不一致就**在性能测量前退出**。
5. **FP8 反量化**：加载器对每个本 rank 拥有的参数做写入覆盖断言，杜绝"静默漏权重"造成的假通过。

## 6. 可观测性与基准方法

**引擎侧**：`LLMEngine.metrics()` 暴露累计 step 数、prefill/decode token 数、调度/前向累计耗时、各 mode（prefill/decode/mixed）step 数；`Scheduler.metrics()` 暴露等待/运行队列长度、抢占次数、KV 块容量/占用/前缀命中 token 数、块分配与回收次数。服务端以 Prometheus counter/gauge/histogram 暴露请求 queue/TTFT/TPOT/E2E 和 step schedule/forward；`--metrics-path` 仍可按 step 输出原始 JSONL。

**基准口径**（对齐社区通行做法，便于横向比较）：

- **离线批处理**：固定 prompt/output 长度、每点先 warmup 再测 3 次，报告吞吐中位数、`sample std / mean` CV、TTFT/请求级 TPOT/token-gap ITL。schema v1 原始结果不回写，schema v2 才使用校准口径。
- **在线开放环**：客户端按 Poisson/等间隔注入请求，必须收到 `[DONE]` 和真实 usage 才记成功；报告 request/input/output/total throughput、TTFT/TPOT/ITL/E2E、最大并发和 SLO goodput，并抓取同窗口引擎计数。
- **指标定义**：TTFT 是到达至首个非空流式输出；请求级 `TPOT=(E2E-TTFT)/(output_tokens-1)`；ITL 是相邻非空流式输出的时间间隔，两者分别统计，不能混名。

参考：nano-vllm（上游引擎结构）、PagedAttention/vLLM（paged KV 与前缀复用）、Orca（iteration-level scheduling）、Sarathi-Serve（stall-free chunked prefill）、vLLM `benchmark_serving`（开放环压测与分位数口径）。

## 7. 已知局限

- SDPA + gather 保留为兼容后端；直接读取 paged KV 的 MUSA Triton decode 后端已按五层门转正为默认（I4，见 `numerics-policy.md`）。
- 未启用 graph（`enforce_eager` 默认），每步有 Python 侧调度与张量构造开销。**捕获本身可行**（30B 六桶全捕、96 次图内 all-reduce 可 replay、TP1 graph==eager bitwise），但 **TP2 引擎级 replay 间歇性楔死**（I29，0.6B TP2 最小复现）→ 整图否决；复活路径 = 分段图或厂商修复（见 ISSUES I29）。
- grouped-MM 的 `argsort`/`bincount`/两次 `_grouped_mm` 调用在小 batch 下固定开销占比高（b1 加速比低于 b16/b32）。
- 在线服务为单实例、单模型、无 tokenizer 批处理；不做多副本与负载均衡。

## 8. P5/P6 优化路线与边界（证据驱动，先冻结阈值再测量）

### 8.1 观测结论（真实 30B，`benchmarks/archive/P5-profiling.md`）

decode 阶段、48 层累计：长上下文 batch 32 时 attention 的 gather+mask 与 SDPA 合计占 forward **58.56%**；MoE 计算最高 **53.48%**（其中 route+combine 最高 44.93%，该数字可能被短段 event 开销放大）；EP all-reduce 最高 **5.45%**；两 rank 路由不平衡最高 **5.19%**。按预先冻结的门槛（30% / 35% 或 10% / 20% / 10%）自动得到：

| 路线 | 门槛 | 实测 | 结论 |
|---|---:|---:|---|
| paged attention | attention ≥30% | 58.56% | 触发，第一优先 |
| MoE fusion | MoE ≥35% 或 route/combine ≥10% | 53.48% / 44.93% | 触发，第二优先 |
| collective overlap | all-reduce ≥20% | 5.45% | **否决** |
| profiled placement | rank 不平衡 >10% | 5.19% | **否决** |

### 8.2 Attention 后端契约

- **prefill 始终用 SDPA**；只有 decode 可在 `sdpa` / `triton_paged` 之间切换（`attention_backend`）。
- Triton paged kernel 的约束：`head_dim=128`、`page_size=256`、GQA 头数可整除；循环上界用**运行时 context_len**（不是 constexpr），因此一份编译服务所有上下文长度，也不会按桶重 JIT。
- 数值：q·k 与 softmax 走 fp32；是否设为默认由 `scripts/p5_decide.py --mode attention` 的冻结门决定（token 全等 + 长上下文 b16/b32 p95 TPOT 比值 ≤0.90 + 吞吐比 ≥1.00 + CV ≤5%）。

### 8.3 MoE 融合边界

- 本项目的 MoE 是 **EP2 + hidden replicated + 一次 all-reduce**：每 rank 只持有 64 个本地专家，非本地专家贡献为 0。
- 允许的实验后端：`loop`（数值参考）、`grouped_mm`（默认）、`fused_grouped_mm`（手写 SiLU+combine 融合）、`vllm_fused`（直接复用安装版 vllm-musa 的 Triton `fused_experts`，opt-in）。
- **不声称**移植 MegaMoE：S4000 无 FP8 算力/SM90 TMA；只借"dispatch→GEMM→act→combine"的融合划分。MoonEP（动态冗余专家）与 UniEP（集群级 EP）依赖多卡 all-to-all，与本项目拓扑不匹配，且实测通信/不平衡都远低于门槛（见 8.1）。
- 实测（git 历史 `docs/P5-moe-optimization-survey.md`）：上游 Triton fused MoE 与 grouped_mm **token 24/24 一致**（bf16 tiny），但默认配置下比 grouped_mm 慢（1024 token 时 262 ms/层）；vllm-musa 的厂商 fast path（mp22）在 1024 token 触发 illegal memory access，故集成时强制走 generic Triton 路径并保持 opt-in。

### 8.4 Graph capture 边界（P6）

- `torch.musa.MUSAGraph` 与 CUDA Graph 等价；仓库已有捕获骨架，但当前 decode 路径不可捕获（`grouped_mm` / Triton kernel 均报 "operation not permitted when stream is capturing"）。
- **MCCL all-reduce 可捕获**（两 rank 数值一致），这是未来重启 P6 的关键前提。
- 重启条件：可捕获的 MoE（或 runtime 修复）+ 形状静态的 attention + 桶裁剪；验收门已冻结在 `scripts/p5_decide.py --mode graph`。

### 8.5 优先级与进入条件

1. **attention A/B**（已实现，待加载）：只看冻结门。
2. **MoE 融合**：只有在提供 `MTT_S4000` 的 autotune 配置（shape-driven，参考 vLLM `get_moe_configs`）后才值得再花加载；否则保持 grouped_mm。
3. **route/pack 融合**：需先用低开销计时复核 44.9% 这个数字（event 插桩可能放大短段开销）。
4. **graph capture**：等 MoE 可捕获后再做，否则收益被 eager MoE 抵消。
