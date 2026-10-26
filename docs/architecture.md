# 架构设计

> 状态：P2 正确性 + 性能、P3 调度、P4 服务化均已实现并通过设备侧回归；数字见 `benchmarks/`。

## 1. 目标与约束

- **硬件**：2× MTT S4000（每卡 48GB、56 SM、768GB/s、PCIe Gen5 x16，无卡间 NVLink/MTLink 实测依赖，TP 通信走 PCIe）。
- **算力限制**：S4000 无 FP8 算力，bf16 为主要计算精度；Triton 存在官方 mtgpu backend（已实测可用）。
- **软件栈**：torch 2.11 + torch_musa 2.11、MCCL 2.11.4（NCCL 等价物，已实测双卡 allreduce）、Triton-mtgpu 3.2、transformers 5.13、Python 3.10。
- **存储限制**：数据盘仅 50GB → 采用官方 FP8 checkpoint（~31.2GB），加载时反量化为 bf16（显存充裕：96GB 总量）。
- **对照组**：实例预装 vllm-musa 0.29.1（含编译好的 MoE C 扩展），作为性能/正确性参照与动机素材（详见 `benchmarks/P0-vllm-musa-baseline.md`）。

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

TP2 通信量：每层 2 次 allreduce（attention 后 + MoE 后）。PCIe Gen5 x16 下隐藏通信不是瓶颈；实测吞吐随 batch 线性扩展（见 `benchmarks/P2-moe-tp2.md`）。

## 5. 正确性策略

分层设门，任何一层失败都不进入性能测量：

1. **逐层数值参考**：`tests/p2_ground_truth.py` 用手写 PyTorch（CPU、无 KV cache、无 TP）对 tiny checkpoint 做贪心生成；tiny 模型的 TP1 输出必须与它逐 token 一致。
2. **并行等价**：`tests/p2_tp_check.py`（TP1==TP2）、`tests/p2_grouped_mm.py`（TP1/TP2 × loop/grouped_mm，外加 round_robin 放置）必须四路 token 完全一致。
3. **解码确定性**：`tests/p2_determinism.py` 反复重建引擎、不做 KV 清零，要求多次结果相同（复现并锁死 MUSA SDPA 读 KV 尾部未初始化数据的 bug）。
4. **30B 端到端**：固定 4 条中英文 prompt 贪心输出连贯；`scripts/p2_benchmark.py --verify-prompts` 会把 token id 与冻结的参考逐 id 比对，不一致就**在性能测量前退出**。
5. **FP8 反量化**：加载器对每个本 rank 拥有的参数做写入覆盖断言，杜绝"静默漏权重"造成的假通过。

## 6. 可观测性与基准方法

**引擎侧**：`LLMEngine.metrics()` 暴露累计 step 数、prefill/decode token 数、调度/前向累计耗时、各 mode（prefill/decode/mixed）step 数；`Scheduler.metrics()` 暴露等待/运行队列长度、抢占次数、KV 块容量/占用/前缀命中 token 数、块分配与回收次数。`--metrics-path` 可按 step 输出 JSONL（mode、批内序列数、query/context 长度、schedule/forward 耗时），用于拆解瓶颈。

**基准口径**（对齐社区通行做法，便于横向比较）：

- **离线批处理**：固定 prompt/output 长度、每点先 warmup 再测 3 次，报告吞吐中位数 + 变异系数（CV）+ TTFT/TPOT p95 中位数。对应 vLLM `benchmark_serving` 的"固定请求集 + 分位数"口径。
- **在线开放环**：客户端按泊松/等间隔注入请求（`--max-qps` × 负载比例），以免闭环压测掩盖排队行为；报告达成 QPS、成功率、TTFT p50/p95、E2E p95，并从 `/metrics` 抓取该负载窗内的 mixed step 数作为调度行为证据。
- **指标定义**：TTFT = 请求到达到首个输出 token；TPOT = 相邻输出 token 间隔（取 p95）。与 Databricks LLM 推理性能工程实践、vLLM 文档一致。

参考：nano-vllm（上游引擎结构）、PagedAttention/vLLM（paged KV 与前缀复用）、Orca（iteration-level scheduling）、Sarathi-Serve（stall-free chunked prefill）、vLLM `benchmark_serving`（开放环压测与分位数口径）。

## 7. 已知局限

- 注意力仍是 SDPA + gather，未接 Triton-mtgpu paged attention kernel；decode 路径每步对所有序列做一次 block gather。
- 未启用 CUDAGraph（`enforce_eager`），每步有 Python 侧调度与张量构造开销。
- grouped-MM 的 `argsort`/`bincount`/两次 `_grouped_mm` 调用在小 batch 下固定开销占比高（b1 加速比低于 b16/b32）。
- 在线服务为单实例、单模型、无 tokenizer 批处理；不做多副本与负载均衡。
