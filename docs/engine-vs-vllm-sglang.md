# 本引擎 vs vLLM / SGLang：做了哪些补充，刻意不做什么

> 目的：回答"除了移植 nano-vllm，这个推理引擎系统有没有向 vLLM/SGLang 靠拢"。
> 口径：nano-vllm 上游约 1.4k 行（continuous batching + paged KV + 前缀缓存 + TP + CUDA graph 骨架）。下表只列**本项目新增或重写**的部分，并标注证据文件。

## 1. 能力矩阵

| 能力 | 本项目 | vLLM | SGLang | 证据 / 说明 |
|---|---|---|---|---|
| Continuous batching | ✅ | ✅ | ✅ | 上游调度器保留 |
| Chunked prefill | ✅ decode-first + 256-token 上限 chunk | ✅ | ✅ | `engine/scheduler.py`（stall_free） |
| 调度策略可选 | ✅ FCFS / stall_free / stall_free_lpm | 多种（优先级/公平） | 多种 | `ScheduledBatch` + 抢占计数 |
| Paged KV cache | ✅ 256-token block | ✅ | ✅ | 上游 `block_manager.py` |
| 前缀缓存 | ✅ **block-hash** + 最长前缀 admission + 饥饿保护 | ✅ hash 前缀缓存 | ✅ **radix tree** | 我们不是 radix tree（已知差距） |
| 抢占（preemption） | ✅ 重算式 | ✅ 重算/交换 | ✅ | 有 `scheduler_preemptions_total` 指标 |
| TP / EP | ✅ TP2 + EP2（attention TP、专家 EP、all-reduce） | ✅ | ✅ | `qwen3_moe.py` |
| MoE 专家后端 | ✅ loop / grouped_mm / fused_grouped_mm / vllm_fused（opt-in） | ✅ 多种 fused kernel | ✅ 多种 fused kernel | 见 `benchmarks/archive/P2-moe-tp2.md` |
| 量化加载 | ✅ FP8(e4m3+block scale)→bf16 反量化（无 FP8 运行时） | ✅ 多种 | ✅ 多种 | S4000 无 FP8 算力 |
| Attention 后端 | ✅ SDPA / Triton paged decode | ✅ FlashAttn/FlashInfer/Triton | ✅ FlashInfer/Triton | `kernels/paged_attention.py`；prefill 仍 SDPA |
| MUSA Graph | ⚠️ 整图否决(I29:TP2 图内 MCCL 间歇楔死);**分段图(图内零 MCCL)离线 b1 −23.7%、跨臂逐位 0 翻转,但在线 A/B 判定 FAIL(并发 14–30 撞桶 padding 税,慢 5.9–14.7%)→ 不转正,默认 eager** | ✅ | ✅ | TP1 graph==eager bitwise;I29/I33;`benchmarks/P11-graph.md`、`P12-serving.md` |
| 请求级指标（TTFT/TPOT/ITL/E2E） | ✅ 每请求返回 | ✅ | ✅ | `benchmarking.py` + `Sequence.metrics()` |
| 引擎/调度/KV 指标 | ✅ + Prometheus histogram | ✅ | ✅ | `prometheus_metrics.py` |
| 步级 trace | ✅ `--metrics-path` JSONL | ✅ | ✅ | 可拆 schedule/forward |
| OpenAI 兼容 API | ✅ chat/completions（SSE、usage、stop、cancel、429） | ✅ 完整 | ✅ 完整 | `entrypoints/openai/api_server.py` |
| 增量 detokenization | ✅ UTF-8 边界 + stop 回退 | ✅ | ✅ | `incremental_detokenizer.py` |
| 异步引擎 | ✅ 后台线程 + 命令队列 | ✅ 异步引擎 | ✅ | `async_engine.py` |
| 采样 | ✅ top_p/top_k/min_p + presence/frequency/repetition penalty + stop_token_ids + logprobs(top-k)（P22;默认参数走逐字 legacy 路径,per-request opt-in) | ✅ top-k/p/min-p/penalty/beam/… | ✅ 同左 | `layers/sampler.py`、`engine/model_runner.py:_build_sample_meta` |
| 结构化输出 | ✅ xgrammar:guided_json/guided_regex/response_format json_schema(P24;per-request opt-in,位掩码与官方 CPU apply 逐位一致,开销 1.028×) | ✅ | ✅ XGrammar | `engine/llm_engine.py:_attach_grammar`、`layers/sampler.py:_apply_grammar` |
| overlap scheduler | ✅ 单飞流水线 collect→postprocess→launch→投机调度(opt-in `scheduler_overlap`;稳态与串行逐位全等;disable 排空式收尾 I38) | ✅ V1 原生 | ✅ 零开销调度 | `engine/llm_engine.py:_step_overlap` |
| 优先级/SLO 分级调度 | ✅ per-request `priority`(高优 TTFT 42.7×;饥饿守卫;抢占 victim 按优先级) | 部分(公平调度,无 per-request 优先级) | 部分 | `engine/scheduler.py:_waiting_candidate/_preempt_victim` |
| 前缀缓存淘汰 | ✅ 两级 victim: fresh 优先 + LRU 冷块 + touch-on-hit(P16-B,默认开) | ✅ | ✅ | `engine/block_manager.py:_pop_free` |
| 结构化输出之外的 tools / n>1 | ❌ 明确不支持（400） | ✅ | ✅ | API 里显式拒绝 |
| 投机解码 / LoRA / 多模态 | ❌ | ✅ | ✅ | 未做 |
| 多实例 / 多节点 | ❌ 单实例单机 | ✅ | ✅ | 未做 |
| PD 分离(prefill-decode) | ✅ 单机双卡 1P1D 全链路:bitwise 交接 + 在线 A/B ITL p95 0.753×/0.839× + 前缀复用 5.7× + OpenAI 端点(P15/P18) | ✅ 多机 | ✅ 一等公民 | `nanovllm/pd/`、`scripts/p15_pd_serve.py` |
| Graph / compile | ⚠️ `torch.compile` 有 opt-in 开关但 MUSA 未验证 | ✅ | ✅ | `utils/compat.py` |
| 评测工具链 | ✅ schema v2 + 开放环 + 冻结决策门 + 可审计 JSON | ✅ `benchmark_serving` | ✅ 自带 bench | `scripts/`、`benchmarks/` |
| 设备侧 profiling | ✅ MUSA Event 分段 + 双 rank + 自动路线决策 | ✅ nsys/torch profiler | ✅ | `profiling.py`、`scripts/p5_profile.py` |

## 2. 相对 nano-vllm 的实质性补充（按子系统）

1. **硬件抽象与移植**：cuda→musa 设备层、MCCL 集合通信、SDPA 后端、Triton-mtgpu 内核；这是"能在国产卡上跑"的前提。
2. **MoE 能力**：Qwen3-MoE 完整实现（softmax top-8 router、EP2、packed experts、grouped GEMM、确定性 top-k 合并）；上游只有 dense。
3. **调度层**：把上游"prefill 优先整步执行"改造成 **decode-first stall-free**，并加入最长前缀匹配 admission 与饥饿保护，同时输出可解释的 `ScheduledBatch`。
4. **可观测性**：从"只有 tqdm"升级为请求级指标 + 引擎/调度/KV 计数 + Prometheus + 步级 JSONL；这是把引擎当系统调优的前提。
5. **服务化**：异步引擎 + OpenAI 兼容协议（含流式、usage、取消、429 背压）+ 增量 detokenizer。
6. **确定性**：修复原子累加导致的贪心不可复现（`index_add_` → slot scatter + 固定顺序求和）、KV 尾部污染、RMSNorm 原位污染。这三条是"能拿来对参考做逐 token 回归"的基础。
7. **实验方法论**：离线/在线双基准、A/B 脚本、**先冻结阈值再测量**的决策 JSON、每次运行带 provenance（commit/模型 sha/参考 sha）。
8. **证据驱动的路线选择**：真实 30B 分段 profiling 直接否决了两条"看起来该做"的路线（通信重叠 5.45%、专家放置 5.19%），避免了盲优化。

## 3. 刻意不做的部分（及理由）

| 不做 | 理由 |
|---|---|
| Radix tree 前缀缓存 | 当前 block-hash 已能命中整块前缀；在单实例、短对话负载下收益不足以抵消复杂度，留作后续。 |
| 投机解码 / LoRA / 多模态 / 分离式部署 | 与"双卡 MoE 推理引擎"主线无关，面试项目里属于堆功能。 |
| 多实例/多节点调度 | 硬件只有 2 卡单机；分布式调度无法验证。 |
| 用 CUDA 生态的 CUDA Graph / cutlass kernel 直接套 | S4000 无对应能力（MUSA graph TP2 replay 间歇楔死 I29、无 FP8 算力），套了也跑不动。 |
| 引入 MoonEP / UniEP / DeepGEMM 源码 | 前提是几十卡 all-to-all 或 SM90+FP8，与 2 卡 replicated-hidden + all-reduce 不匹配；只借鉴融合划分。 |

## 4. 结论

- 它已经不是"nano-vllm 换设备"：调度、指标、服务化、确定性与评测方法都补齐到了**可解释、可复现、可对比**的程度。
- 与 vLLM/SGLang 的真实差距集中在：graph capture（硬件阻塞）、投机解码（判负）、量化推理运行时（平台 fp8 no-op，见 ISSUES I20 终判）、radix 前缀缓存、多实例与多节点、以及自定义 kernel 的成熟度（我们的 paged/MoE kernel 还在"正确 + 局部更快"阶段）。
- P22–P24 已把服务面对齐（采样面/overlap 调度/结构化输出），P25 评审轮修复了 overlap 的两个状态机缺陷（ISSUES I37/I38）并为运行时开关补了服务端 CLI（`--scheduler-overlap`）。当前差距的量化比对与求职视角评估见 `docs/P25-proposal-and-career-review.md`。
