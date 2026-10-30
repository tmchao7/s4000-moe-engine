# 项目成果与面试材料（RESUME）

> 用途：把本仓库的实测结论整理成可直接用于简历/面试的表述，并明确哪些数字**能写**、哪些**必须附口径**、哪些**不能写**。
> 口径原则：只列有冻结门、有对照、有证据文件的数字；跨轮对比不作数（`benchmarks/P12-serving.md` 明确）；负结果照样是结论。
> 证据路径均相对仓库根。数字的最终权威是代码与实测，冲突时以证据文件为准。

## 0. 一句话版本

在国产 GPU（摩尔线程 MTT S4000 ×2）上，把 1.4k 行的 nano-vllm 移植并扩展成能服务 Qwen3-30B-A3B-FP8 的 MoE 推理引擎（TP2/EP2、MUSA 全栈、OpenAI 兼容），并用**预注册冻结门**的 A/B 方法逐项判定性能改动——其中「分段图捕获」把 30B 单流 decode 步时从 152.85 ms 降到 116.63 ms（**−23.7%**），同时保持逐 token 数值等价。

## 1. 背景与硬约束（面试先讲这个，能立住"为什么难"）

| 项 | 事实 | 为什么重要 |
|---|---|---|
| 硬件 | 2× MTT S4000，48 GB/卡，PCIe Gen5 | 显存紧：30B bf16 权重 29.5 GiB/卡，KV 只能分到 ~8.5 GiB |
| 算力缺陷 | **S4000 无 FP8 计算单元** | FP8 权重必须加载时反量化到 bf16；官方 vLLM-MUSA 反量化 >30 min 不可用 → 自研加载器 |
| 软件栈 | torch_musa 2.11 + MCCL + Triton(mtgpu) | MCCL 在 graph 内会间歇性 rendezvous 楔死（I29），直接封死"整图"路线 |
| 模型 | Qwen3-30B-A3B-FP8，30.5B 总参/3.3B 激活，128 专家 top-8 | MoE 的 all-to-all/专家并行在国产卡上要自己搭 pack + EP2 |
| 基线 | 官方 vLLM-MUSA 双卡 **8.4 tok/s** @c32，吞吐不随并发扩展 | 瓶颈是每步固定开销，这就是整个项目的优化靶心 |
| 起点 | 上游 nano-vllm ~1.4k 行，CUDA 生态 | 不是改配置，是把设备层/通信/内核全换掉 |

## 2. 我做了什么（四块，各带数字）

1. **MUSA 移植**：device 抽象层、MCCL 集合通信、SDPA、Triton-mtgpu paged attention kernel。
   结果：首 token 与 HF 参考 **5/5 = 100%** 一致；单卡 64 并发 **113.48 tok/s = 官方双卡基线的 13.5×**。
2. **MoE 支持与加速**：Qwen3-MoE 完整实现（top-8 router、EP2 packed experts、确定性合并），三种 expert 后端可切换。
   结果：`grouped_mm` 比朴素 loop **3.0–3.2×**；自研 `bmm_dense` decode 后端比 grouped_mm **1.44–1.53×**（b16）/ **1.29–1.37×**（b32）；离线 **141.48 tok/s** @b64（CV 0.0028）。
3. **服务化与可观测**：OpenAI 兼容流式 API（SSE/stop/cancel/429）、请求级 TTFT/TPOT/ITL、双口径 SLO、Wilson CI、Prometheus、步级 JSONL trace、decode-first `stall_free` 混步调度、chunked prefill。
   结果：混步调度把 mixed step 从 0 拉到 124/轮、prefill step 131→4；chunk-2048 把 1k 输入 50% 档 **TTFT p50 3.31 s → 1.036 s**；前缀缓存 **TTFT p95 −25.6%**；paged attention 按五层等价门转正，**TPOT p95 −12.1/−16.7/−17.7/−18.4%**（四档）。
4. **分段图捕获（本项目最硬的一块）**：整图在 TP2 上会间歇性楔死（用分级探针定性为 MCCL-in-graph，给出 0.6B 最小复现）；改为**每层切两段、把全部 all-reduce 挪到图外 eager 执行**的分段捕获。
   结果：30B 双卡 **768 张图/rank 捕获零失败（5.2 s）**，显存 **+0.00 GiB**，跨臂 **逐 token 0 翻转**；**b1 decode 152.85 → 116.63 ms（−23.7%，1.31×）**；0.6B 上 **−47.2%**（55.85 → 29.50 ms）。据此进一步定位到「单流 decode 步时约 36 ms/步是逐算子 host launch 开销，replay 可回收」。
   **然后在线 A/B 把它否了**（预注册三门）：四档 TPOT p95 比 eager 慢 **5.9–14.7%**，同 session 复跑确认、排除机时污染。原因定位为**桶化捕获的 padding 税**——桶集 `[1,2,4,8,16,32,48,64]` 整桶 replay，在线并发 14–30 落在桶之间，被 pad 的行照算 attention/MoE；而离线门只测 b1/b8/b32（恰好全在桶边界），结构性看不见。结论：**该优化只对单流成立，不进默认**。这块的价值不在"提速了"，而在"用自己的门否掉了自己的优化，并说清为什么"。

## 3. 可以直接写进简历的数字（A 档）

| 数字 | 对照 | 冻结门 | 证据 |
|---|---|---|---|
| 30B b1 decode **−23.7%**（152.85→116.63 ms，1.31×） | eager vs segment 各 3 跑，CV ≤3.13%/0.31%；跨臂 token 逐位 0 翻转 | b1 ≥+15% + token 全等 + 显存 ≤+2 GiB，五项全过 | `benchmarks/P11-graph.md`、`benchmarks/data/p11/p11_30b_*` |
| 0.6B b1 decode **−47.2%**（55.85→29.50 ms） | 同上双臂交错 | ≥−8% 门 | `benchmarks/P11-graph.md` |
| 30B 双卡图捕获 **768 图/rank 零失败、5.2 s、显存 +0.00 GiB** | 整图路线在 TP2 楔死（I29）作对照 | 零楔死 + bitwise | `benchmarks/P12-serving.md`、`benchmarks/P11-graph.md` |
| paged attention 转正：**TPOT p95 −12.1/−16.7/−17.7/−18.4%** | sdpa vs triton_paged | 五层门 L1–L5 全过（翻转率 0.213% < 0.5%、首 token 100%、12/12 人工判读） | `docs/ISSUES.md` I4、`benchmarks/P6-serving-v4.md` |
| 离线 **141.48 tok/s**（b64，CV 0.0028） | vs 上版默认 111.75 → **+26.6%** | 已关闭（I5） | `benchmarks/P6-i5-i6-i7.md` §1 |
| 前缀缓存 **TTFT p95 −25.6%**（1.256→0.934 s） | 无缓存基线 | 已关闭（I6） | 同上 §2 |
| prefill host-sync 上提 **−11.6%**（0.6B 512-token 62.8→55.5 ms） | 同实验 A/B | 已修复（I24） | `docs/ISSUES.md` I24 |
| chunk-2048：1k 输入 50% 档 **TTFT p50 3.31→1.036 s** | 同矩阵旧值 | ≤2 s 门 PASS | `benchmarks/P9-nightly.md` |
| 单卡 **113.48 tok/s = 官方双卡 13.5×** | 官方 vLLM-MUSA 8.4 tok/s | 首 token 100% 对齐 | `benchmarks/archive/P1-musa-port.md` |
| `bmm_dense` decode 后端 **1.44–1.53×**（b16） | grouped_mm | 吞吐比 ≥1.08 + CV ≤5% | `docs/ISSUES.md`、`benchmarks/archive/P6-nightly.md` |
| 投机解码 **NO-GO**：α=0.0056、c=0.384、最优 k=1 仅 **0.725×** | 32 prompt 实测 vs 冻结门 1.3× | S1 判决 | `benchmarks/archive/P7-phase0.md` |
| **生产栈容量下界 ≥1.070 req/s**（137-token/128-output，1.6× 过载仍双口径 100% SLO，并发 57，拐点未探到） | I7 旧栈 0.94 / 段图臂 1.036 | 三 repeats 全 ≥99% | `benchmarks/P13-capacity.md` |
| **1k 长上下文：冷缓存塌方机理与解法**（chunk-1024 砍冷 TTFT p50 −56~57%，0.5 档冷态 90.6% SLO 过门；准入修复 1536+ 请求零 429 零崩） | chunk-256 冷态全档塌（TTFT 3.3–5.1 s） | ≤2 s 门 | `benchmarks/P13-longctx.md` |

## 4. 必须附口径才能写（B 档）

| 数字 | 口径怎么说 |
|---|---|
| 臂 B′ 四档 **slo_standard 全 100%**、TPOT p95 中位 **0.211/0.270/0.272/0.285 s**、饱和 1.4× 档（目标 1.218 qps）仍 100%、实测 **1.036 req/s** | 写「段图形态服务的绝对能力」，**不要写成"相对 eager 提升"**；A/B 判定见 §3 与 `benchmarks/P12-serving.md`。饱和拐点上界未探到，只能说 ≥1.218 qps |
| 30B b8 持平（+0.5%）、b32 **+3.8% 回退** | 必须如实写「高载档真实回退」，这是 b32 走 device 饱和区的代价 |
| I19：「每步 2,401 次 launch、平均 13.5 µs、device_busy 0.478–0.603」 | 是有对照的诊断结论，写成"性能诊断能力"可以；别写成已落地的优化 |
| 长上下文：1k 输入容量在 0.435 QPS 崩、三档 SLO 0% | 必须同时说"瓶颈在 TTFT 不在 decode"，否则看起来像失败 |

## 5. 不要写（C 档）

- 「分段图让在线 TPOT 降了 27%」这类**跨轮对比**——实证是**反向**的：同轮 A/B 里段图慢 5.9–14.7%，判定 FAIL。
- 「分段图已上生产/已默认」——默认仍是 `enforce_eager=True`、段图 env 默认关。
- 「整图 graph 快 1.55×」——整图在 TP2 被否决（I29），默认仍 eager。
- 「TPOT p95 0.327 s 是当前栈」——那是 v4 单跑，I30 明确质疑跨轮可比性。
- 「chunk-2048 已默认」「关闭 mixed-split」——两个默认化改动都被冻结门否决。
- 「长上下文 SLO 达标」——实测 0%。

## 6. 负结果清单（面试里是加分项：证明会否决自己的优化）

| 被否决的方案 | 否决依据 | 说明 |
|---|---|---|
| **分段图在线转正** | 在线四档 TPOT p95 比 eager 慢 5.9–14.7%，门②③ FAIL；同 session 复跑确认 | 离线 b1 −23.7% 不能外推到并发场景，根因是桶化捕获的 padding 税（详见 §2 第 4 条） |
| 投机解码（朴素 draft） | α≈0.006、c≈0.385，最优 k=1 **0.73×**，距 1.3× 门差 44% | 两个独立原因：draft 步太慢（每步固定开销）+ 跨代模型首 token 即分叉 |
| 整图 CUDA/MUSA graph | TP2 引擎级间歇 wedge（0.6B 最小复现 3/3），barrier 无效 | 分级探针把"图内 MCCL"定位为必要成分 → 才有分段图这条路 |
| chunk-2048 转默认 | TPOT p95 **+5.3~9.5%**，超 +5% 门 | 只保留为长上下文显式参数 |
| `NANOVLLM_MIXED_SPLIT=0` | 低载无收益（与基线重合），高载 SLO 崩到 **22.7%** | 混步拆分的假设被否 |
| fused RMSNorm | **−0.76%**（且 launch 数 −960/步、device −10.7 ms 但步时不降） | 直接否掉"launch-bound → 融合换步时"的因果链 |
| profiled expert placement | 负载不均衡 **−99.5%** 但吞吐 **+0.01%** | 动机被证伪，不采用 |

## 7. 方法论（可讲的工程素养）

1. **先写门再跑**：每个实验在跑之前把判据、容差、判定树写进报告，跑完不改阈值（`benchmarks/P12-serving.md` 的三道门就是范例）。
2. **五字段报告**：门 / 口径 / 结果 / 判定 / 证据。判定只有三种：PASS、FAIL、未产出——**缺数据就不出判定**（P12 臂 A′ 缺损时明确写"未产出，不伪造判定"）。
3. **A/B 噪声协议**：交错 ≥5 次取中位数、CV ≤5% 否则判 INVALID；在线侧用 3 repeats 中位 + Wilson CI + 每请求 records 复算双口径 SLO。
4. **数值等价分层**：结构（max_abs）→ 翻转率/margin → 质量（首 token/人工判读）→ 确定性 → 在线 SLO，五层全过才转正。
5. **必须丢首 batch**：30B 进程内首个 batch 与稳态逐位不同（一次性 warmup 态跃迁，I31）；不丢就把"预热"误判成"数值漂移"。
6. **负结果照录**：上面 6 条被否决方案全部留在台账里，附现场日志与崩溃形态。

## 8. 高频追问与应答要点

1. **"为什么不用整图，要用分段图？"**
   整图在 TP2 上会间歇性 rendezvous 楔死；分级探针显示纯张量图（含 96 次图内 all-reduce）全绿、TP1 从不楔死、0.6B TP2 paged 首次 replay 3/3 楔死 → 必要成分是"引擎 replay 路径 × TP2 集合通信"。分段图把每层切两段、所有 all-reduce 放到图外 eager 执行，图内零 MCCL，于是拿到 bitwise 正确性。
2. **"为什么 b8 持平、b32 反而回退 3.8%？在线为什么更差？"**
   分段图省的是 host launch；b8 以后已进入 device 饱和区，且每层有 2 次 eager all-reduce，省下的发射时间被 AR 与设备计算抵平甚至倒贴。在线还有第二层原因：图是按桶捕获的（`[1,2,4,8,16,32,48,64]`），replay 取"第一个 ≥ 当前 batch 的桶"并整桶执行，所以 batch 14 会按 16 跑、21–27 按 32 跑——被 pad 的行数值上无害（slot 指向 scratch 块），但计算是实打实的。在线并发正好是 14–30，落在桶与桶之间，于是净负 5.9–14.7%。
3. **"你们容量到底多少？先崩的是哪个指标？"**
   短 prompt（137 token）0.87 qps 四档全 100% SLO；压到 1.4×（1.218 qps）仍 100%。长上下文（1k token）在 0.435 qps 就崩，而且**崩的是 TTFT 不是 TPOT**——单请求能力正常（空闲 TTFT 0.368 s），崩在 1k prefill 与 decode 混跑的排队，所以才有准入/背压和 chunked prefill。
4. **"I30 是什么，为什么还开着？"**
   现行栈的短 prompt TPOT p95 比 v4 时期高 13~16%（0.387 vs 0.343 门），跨 repeats 稳定、不是噪声。已排除 mixed-split 税；剩余两个候选是 P8 期累积改动二分、以及跨轮口径复核（v4 那个 0.327 是单跑）。按冻结门它仍然开着——不因为我们喜欢，而是没证伪。
5. **"你怎么保证优化没改数值？"**
   五层等价门 + 跨臂逐位比较。图这条线最终证据是：丢弃首 batch 后，跨臂 eager vs segment 在 b1/b8/b32 × 64 token 上**逐位 0 翻转**。
6. **"为什么生产默认还是 eager？"**
   因为在线 A/B 判定是 **FAIL**，按冻结门不翻默认——这是项目红线（先判定再改默认，判定不改阈值）。另外翻默认本身还带两个硬前提：`--decode-moe-backend bmm_dense`（否则 decode 落到 grouped_mm，不是生产栈）和 `max_num_seqs ≤ 64`（否则 30B 捕获期 3360 张图 OOM）。
7. **"最大的性能杠杆是什么？"**
   单流 decode 的每步固定开销。实测 0.6B 每步 2,401 次 kernel launch、平均 13.5 µs、device_busy 0.478–0.603，且 0.6B 与 1.7B 的 b1 步时几乎相同 → 是 CPU 发射在关键路径上。分段 replay 回收其中 36.2 ms/步；纯 eager 路径上"融合小算子"被实测否掉（−0.76%）。
8. **"S4000 上最大的坑？"**
   没有 FP8 算力，只能加载时反量化（351 s，冷页缓存 11–12 min）；MUSAGraph 在 TP2 下与 MCCL 组合不稳定；驱动楔死后 10 分钟才被杀，报错点还会漂移（所以自己写了 watchdog 抓现场）。
9. **"如果给你更多机器/时间，下一步做什么？"**
   KV 量化（KV 目前 bf16 12 MiB/block，容量 ×2 的量级）、b32 回退归因（AR/replay 重叠或高载自动降级 eager）、1k prefill 的 kernel profile、以及把 I30 的二分做完。
10. **"你的优化有没有翻车过？怎么发现的？"**
    有，而且这条线最后成了项目里最完整的一次闭环。分段图离线 b1 −23.7%，但在线 A/B 四档 TPOT p95 全线慢 5.9–14.7%——预注册的门直接判 FAIL。我做了三步：①同 session 复跑排除机时污染（复现到 0.1%）；②定位到桶 padding（图按 `[1,2,4,8,16,32,48,64]` 整桶 replay，在线并发 14–30 落在桶间），并发现离线门只测 b1/b8/b32 这些桶边界、结构性地看不见它；③**动手修**：照 vLLM 的密桶梯子（11 桶、1056 图/rank 零失败）+ SGLang 的"padding 即回退 eager"语义，同 session 再测——padding 税被证实并消除（0.9 档 +14.7%→+2.3%），但残余 +4.7% 超过我们设的 +2% 门，残余恰好等于 P11 在桶边界测到的固有开销（每层 2 次 eager all-reduce 的串行化，属于整图范畴、已被 I29 否决）。最终按门关闭分支：段图定格为单流开关。整个过程：预注册门 → 发现微基准收益不能外推 → 修复 → 再测 → 按门收口，每一步都有数据。

## 9. 简历 bullet 草稿

**中文（4 条，按需删减）**

- 在国产 GPU（摩尔线程 MTT S4000 ×2）上基于 nano-vllm 构建 Qwen3-30B-A3B-FP8 的 MoE 推理引擎：完成 MUSA 设备层/MCCL/Triton 内核移植，实现 TP2+EP2 packed experts；单卡 64 并发 113.5 tok/s，为官方 vLLM-MUSA 双卡基线的 13.5×。
- 设计并实现**分段图捕获**：定位整图在 TP2 下 MCCL 图内集成导致的间歇性楔死（分级探针 + 0.6B 最小复现 3/3），改为每层两段、all-reduce 移出图的分段捕获——30B 双卡 768 图/rank 零失败、显存零增量、跨臂逐 token 0 翻转，单流 decode 步时 152.85→116.63 ms（**−23.7%**）；再用预注册的在线 A/B 证明该收益只存在于单流（并发 14–30 因桶 padding 净负 5.9–14.7%，B 臂同 session 复跑确认），据此**否决转正**、默认保持 eager。
- 搭建性能实验体系：预注册冻结门 + 五字段报告 + 交错 A/B（CV≤5%）+ 请求级双口径 SLO（Wilson CI）；据此转正 paged attention（TPOT p95 −12~18%）、bmm_dense decode 后端（1.44–1.53×）、前缀缓存（TTFT p95 −25.6%），并**否决** chunk-2048 默认化、fused RMSNorm、profiled placement 等 6 个方案。
- OpenAI 兼容服务化与可观测性：SSE 流式、请求级 TTFT/TPOT/ITL、Prometheus、步级 trace、decode-first 混步调度与 chunked prefill；1k 输入 TTFT p50 3.31→1.036 s，短 prompt 0.87 qps 四档 100% SLO，过载至 1.4× 仍 100%。

**English (3 bullets)**

- Ported and extended nano-vllm into a production MoE inference engine for Qwen3-30B-A3B-FP8 on dual Moore Threads S4000 GPUs (MUSA/MCCL/Triton), with TP2+EP2 packed experts; reached 113.5 tok/s at 64 concurrency — 13.5× the vendor vLLM-MUSA dual-GPU baseline.
- Designed **segmented graph capture** for TP2 MoE decoding: diagnosed intermittent MCCL-in-graph rendezvous deadlocks with a graded probe (minimal 0.6B repro 3/3), then split each layer into two captured segments with all-reduce kept eager — 768 graphs/rank with zero capture failures, no memory overhead, bitwise-identical tokens across arms, and b1 decode step time 152.85 → 116.63 ms (**−23.7%**). A pre-registered online A/B then showed the win exists only for single-stream (5.9–14.7% slower at concurrency 14–30 due to bucket padding, replicated same-session), so I **rejected the rollout** and kept eager as the default.
- Built a pre-registered gating methodology (frozen thresholds, five-field reports, interleaved A/B with CV≤5%, per-request dual-SLO with Wilson CIs) that adopted paged attention (TPOT p95 −12–18%), bmm_dense decode (1.44–1.53×) and prefix caching (TTFT p95 −25.6%), while explicitly rejecting six optimizations including chunk-2048 default-on and fused RMSNorm.

## 10. 一句话解释"我没做但知道该怎么做"

投机解码、KV 量化、EAGLE3/PARD、radix 前缀树、多实例/分离式部署——台账 `docs/ISSUES.md` 里都有条目、有前置条件、有判定树，但都没开工。面试被问到可以直说范围，不要含糊成"做了"。
