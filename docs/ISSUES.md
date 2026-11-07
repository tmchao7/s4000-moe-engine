# ISSUES — 全项目问题台账（唯一清单）

> **本文档是问题清单的唯一权威**（滚动更新,无日期后缀;历史版本与被取代的 `P6-open-issues.md` 见 git 历史）。
> 计划与冻结门见 [`docs/PLAN.md`](./PLAN.md);当前状态见 [`docs/HANDOFF.md`](./HANDOFF.md);历史报告在 `benchmarks/archive/`。
> 规则：**文档与代码/实测冲突时以代码和实测为准**；每个实验先写门再跑；失败/负结果照样登记。

## 0. 状态速览

**编号规则（三系列，互不重号）**

| 系列 | 范围 | 编号 |
|---|---|---|
| **I** | P6 遗留技术债（引擎/图/数值/验证） | I1–I18（沿用历史编号，I16–I18 已迁入 E 系列，见 §3） |
| **S** | P7 投机解码 | S1–S6 |
| **E** | 生态与厂商（autotune 手搓边界、runtime 限制） | E1–E6 |

**统计（P15 后：I 31 项 = 已关闭 19 + 未结 12（新增 I34）；S 6 项；E 6 项。P15 给 I29 增补新栈证据（维持）；I28 变体已工程化修入 nanovllm/pd/kv_channel.py。I12 引擎侧兜底已落地，仍留 🟢 观察）**

| 系列 | 已关闭 | 未结 | 未结条目 |
|---|---:|---:|---|
| I 技术债 | 19 | 12 | I1、I2、I3、I12、I14、**I19**、I20、I27、I28、**I31**、**I32**、**I34** |
| S 投机解码 | 1 | 5 | S2–S6（S5 随 S1 搁置） |
| E 生态厂商 | 0 | 6 | E1、E2（可手搓）、E3–E6（厂商侧） |

已关闭 20 项（I 19 + S1）：I4、I5、I6、I7、I8、I9、I10、I11、I13、I15、I21、I22、I23、I24（部分修）、I25、I26、**I29、I30、I33**、S1。
> 编号：I16–I18 已迁入 E 系列（E2/E5/E6），不再单独计数，故 I 系列实际 30 项（I1–I15、I19–I33）。
> I8 已用有效跑闭合（1.0–1.15k token、384/384、rank1 存活）；三档 SLO 0% 是真实容量结论（瓶颈在 TTFT），P13 补齐机理与解法（见 I8 条目 P13 终局）。

状态图例：🔴 阻塞主路径／决定立项　🟡 收益被冻结或覆盖缺口　🟢 工程债/低优　⚪ 厂商 runtime，我们不可修

---

## 1. P6 技术债（I 系列）

### I1 🔴 图路径控制组全失败；禁止 30B 图 A/B
- **现状**：历史 30B 第一次 replay 会楔死；按固定 `SDPA + bmm_dense decode` 重做 TP2 隔离，tiny MoE 与 Qwen3-8B dense 均 **0/5**（tiny 是稳定 token 漂移；8B 五次均超 120 s watchdog）。无单一可证实的代码级根因，30B graph A/B 被前置门阻止。
- **证据**：`benchmarks/data/p6b/p6_graph_isolation.json/.jsonl`（`80832dd809500274…` / `096fe57cd83ebe645…`）；历史 30B 现场 `p5_graph_ab.log`。旧"8B 5/5 绿"只适用于当时变量组合，不可外推。
- **已排除**：rank 间 replay 错位（8B 上 barrier 有效）；跨桶 pool 复用（8B 上私有池有效）——两者在 30B 都不够。
- **候选路径（当前一律不做）**：① 分段捕获（attention/linear 进图、MoE eager，SGLang 式）；② 图内去 MCCL；③ 厂商 MUSAGraph 规模/依赖链限制清单；④ 8B+MoE 中间档隔离变量。
  **前置条件**：仅在拿到厂商 runtime 限制/诊断证据或单一可验证代码根因后才允许重开；在此之前不做猜测性 patch、不做 30B A/B、不烧加载。
- **参考**：git 历史 `docs/P6-musa-graph-survey.md`（上游如何用"按桶捕获+padding"处理 MoE 图；摩尔 vLLM 分叉在 S4000 上默认对 MoE 开 MUSA Graph——这条线索值得向厂商举证）。
- **影响**：decode 图 1.1–1.2× 拿不到（eager 已由 bmm_dense 拿到大头，图增量本就变小）。

### I2 🟡 图 replay 首轮即漂移（已定界，与 I1 同源）
- **现状**：graph→eager→graph、三次连续 graph、清/不清 prefix cache 均不能消除；`MUSA_LAUNCH_BLOCKING=1` 重放下仍见首个 graph replay 漂移。历史上 8B 曾出现 gen1→gen2 一条 prompt token 变化（`[353,353,353,353]`→`[353,10,4999,1725]`），gen2==gen3。
- **证据**：`benchmarks/data/p6b/p6_graph_3gen_bisect12.log`、`p6_graph_isolation.json`。
- **结论**：不是 I4 的 Triton 数值路径，也不能靠清 prefix cache 解决；**即使 I1 修好，输出一致性是图转正的第二道门**。
- **下一步**：保持 eager；只等 I1 的根因证据。

### I3 🟡 tiny 图捕获间歇失败（本轮未复现，不关闭）
- **现状**：本轮 5-run 隔离的 tiny 捕获均完成，失败发生在后续一致性门而非 capture；但既有 `operation not permitted` 现场（`reg_bmmdense.log`）不能被"一次未见"否定。
- **影响**：图实验样本量受捕获噪声污染；捕获失败与 replay 漂移必须分开计数。
- **下一步**：失败即独立进程重试 + 捕获失败时自动加 `MUSA_LAUNCH_BLOCKING=1` 定位 operator；把捕获成功率作为图实验固定前置指标。

### I4 ✅ 已转正（10-27 19:00）：L1–L5 全过，默认已切 `triton_paged`
- **诊断（已完成，不再重做）**：差异是 **bf16 归约顺序**，非页表/mask/tail bug。同状态 teacher-forced probe：8,928 request-step 中 **19 次 top-1 翻转（0.213%）**，翻转处 top-2 margin **全部 ≤0.3125**；页边界/乱序页/poison-tail 参考误差 `1.22e-4 < 2e-3`。
- **门已按业界标准重写**（`docs/P6-numerics-policy.md`；实现 `scripts/p6_decide_attention_equivalence.py`；旧 `p5_decide.py` 阈值**未改**，其 `keep_sdpa` 作为历史保留）：

  | 层 | 结果 |
  |---|---|
  | L1 结构 | ✅ `max_abs=1.22e-4` |
  | L2 数值 | ✅ 翻转率 0.213% < 0.5%；margin 0.3125 < 0.5 |
  | L3 质量 | ✅ 首 token 一致率 **100%**、退化 0/0、**12 组人工判读 12/12 语义等价** |
  | L4 确定性 | ✅ 两后端各 3 次重复逐 token 一致（两次独立实验数值完全相同） |
  | L5 性能 | ✅ **四档 100% SLO**；100% 档 TTFT p95 **1.203 s**、TPOT p95 **0.327 s**、goodput **0.776**；TPOT p95 −12.1%/−16.7%/−17.7%/−18.4%。所谓"启动挂死"实为 FP8 逐 expert 反量化 351 s，到 `/health` 共 377 s |

- **L3 的关键发现（推翻了初版门）**：初版用字符级编辑距离 p95 ≤8% → 实测 **p95 65%、仅 20/100 条请求完全一致**，看似"质量崩了"。但 **LCP 中位 42.5 / 均值 60.7（共 128）** + 首 token 100% 一致 → 现象是"先一致 1/3~1/2，遇一次近平局翻转后整段走不同分支"。12 组人工判读全部语义等价（例："降低显存/提高速度/降低功耗" vs "计算量/内存/延迟"）。**编辑距离度量的是措辞，不是质量**，故 L3 改为"首 token + 退化 + 人工判读"。
- **当前决策**：`benchmarks/p6-attention-equivalence-decision.json` = **`adopt_triton_paged`**，L5 已过；`nanovllm/config.py:33` 与 `api_server.py` 默认值已改为 `triton_paged`，sdpa 保留为显式 opt-in 与历史参考后端。
- **证据**：`benchmarks/P6-serving-v4.md`、`benchmarks/data/p6b/p6_attention_equivalence.json`、`p6_L3_review.json`、`p6_attention_logits_i4.json`、`benchmarks/P6-L3-review.md`
### I8 ✅ 已闭合（10-27 19:42，重设计后有效跑）：长上下文可服务，但容量在 0.435 QPS 就崩
- **有效跑**：`prompt_mode=unique`、`--prompt-tokens 1024`（tokenizer 校准 68 词 → 1,018 token；逐请求实际 1,018–1,154）、`--max-model-len 4096`、三档 0.5/0.75/1.0、128 请求/档、64 token 输出；栈 = 新默认（paged + bmm_dense + stall_free）。
- **有效性**：384/384 成功且 `[DONE]`；5 s watchdog 全程 rank1 存活（只有 spawn 前第一条为空）；长度门按字面未过（二/三档 p50 1,154，超 ±6% 到 +12.7%），但落在 0.99–1.13× 目标区间，按"长度区间有效"使用并在报告中注明。
- **结果**：三档 SLO **全部 0%**——TTFT p50 **3.84 / 8.28 / 10.22 s**、p95 5.48 / 12.45 / 14.65 s；TPOT p50 0.418 / 1.175 / 1.419 s；最大并发 **25 / 80 / 104**（v4 同请求率是 14/23/24）。
- **失败定位**：**TTFT，不是 decode**。50% 档满足 TTFT≤2 s 的请求 0/128，满足 TPOT≤0.5 s 的 83/128；空闲时同长度单请求 TTFT **0.368 s**、TPOT **0.144 s**。→ 单请求能力正常，崩在"1k prefill 与 decode 混跑 + 无准入控制"的排队。
- **与上轮无效跑的区别**：上轮 18,354 token、8/128 完成、rank1 死亡；本轮 1.0–1.15k token、384/384、rank1 存活。**这次的 0% SLO 是真实容量结论，不是故障。**
- **下一步**：I26 准入/背压 → I23 mixed step 后端 → prefill chunk 曲线 → 重定长上下文目标 QPS（0.87 是 137-token 口径）。
- **证据**：`benchmarks/P6-i8-long-context.md`、`benchmarks/data/p6d/p6_i8_long_unique_v2.json`（sha256 `1249e49bcb3de838500bf0e0e9007310597ee27957deddb0c54017cfacaa297e`）、`i8_watch.log`（`dd3b0c1e…`）
- **P13 终局（10-30，机理+解法闭合）**：I26 准入在 1k 负载端到端验证（1536+ 请求零 429、零崩、无雪崩）；**冷缓存 1k 塌方的机理 = chunk 粒度**——chunk-256 冷 TTFT p50 3.30/4.14/5.07 s 全档塌，chunk-1024 砍到 1.449/1.784/2.201 s（−56~57%），0.5 档冷态即过 ≤2 s 门（90.6% SLO）。**⚠️ 口径发现**：跨 repeat 的 unique prompt 确定性重建 → rep2/3 全命中前缀缓存，"三 repeat 中位"实测的是暖缓存路径（rep2/3 两臂均 100%）；冷容量必须看 rep1 或改 seed-per-repeat。**P14 已根治客户端侧**：`benchmark_online.py --prompt-seed-per-repeat`（默认关=行为不变；开启后 repeat 1 沿用 `--seed`，repeat k 用 `seed+(k-1)*1000003`，各 repeat prompt 串与首词全不相同；实际 seed 写入结果 JSON 的 `prompt_seeds_used`），单测 `test_prompt_seed_per_repeat_changes_unique_prompts`。**P14 实测**：开 seed-per-repeat 的 1k 冷态跑三 repeat 前缀命中增量均为 0（真冷）；chunk-2048 冷态 1.0 档 TTFT p50 1.08–1.12 s **过 ≤2 s 门**（chunk-1024 为 2.2 s），代价是最大 ITL p95 ~1.2–1.5 s、slo_strict 仅 18–24%（见 `benchmarks/archive/P14-serving.md`）。推荐配置：1k 场景 chunk=1024（冷）/前缀缓存（暖）/准入（稳定）三件套；chunk-256 短 prompt 默认不动（P10 门仍有效）。证据：`benchmarks/archive/P13-longctx.md`。

### I12 🟢 崩溃进程 teardown 卡死（P14：引擎侧兜底已落地）
- **P14 兜底**：`LLMEngine.exit()` 加看门狗线程（`NANOVLLM_EXIT_TIMEOUT_S`，默认 60 s，0=关闭）：超时则 kill 全部 rank 子进程、unlink `/dev/shm/<shm_name>`、`os._exit(0)`；正常退出路径零影响。单测 `tests/p14_exit_watchdog.py`（hang/normal/disabled 三例 PASS）。**未覆盖**：rank0 主进程自身卡在 CUDA/MUSA 驱动调用内（线程仍可触发 `os._exit`，但设备侧残留需 `mthreads-gmi` 核验）；SOP 的按 PID 清理仍保留。
- **现状**：rank 间 DistBackendError 后主进程挂在退出路径（bmmdense 回归、30B graph A/B 均遇到），需人工按 PID kill。
- **下一步**：engine `exit()` 加 barrier 超时 + `os._exit` 兜底；短期靠 SOP（AGENTS.md 红线节）。
- **10-27 第三类现场已澄清（不是挂死）**：`api_server --attention-backend triton_paged` 启动时两 rank 各占 29.5 GiB、零日志、0% util、`/health` 无响应，曾被记为"启动挂死"。加 `NANOVLLM_TRACE_INIT=1` + `faulthandler` 后实测：`load_model` 的 FP8 逐 expert 反量化占 **351.4 s（rank0）/354.0 s（rank1）**，warmup 16 s，KV 分配 0.08 s，**到 `/health` 共 377 s**；faulthandler 栈全部落在 `loader.py:103/108`。冷页缓存时该段 11–12 min（本轮参考基线生成实测）。**与 paged 无关**：warmup 只跑 prefill（恒 SDPA），paged 模块首次 decode 才 import，`allocate_kv_cache` 无 backend 分支。**运维修正**：启动窗口按 ≥20 min（冷）设置；判活用 `NANOVLLM_TRACE_INIT=1`，不要用"显存已占"推断"权重已加载完"。
- **10-27 另一个 SOP 漏洞**：rank1 的 spawn 子进程 argv 是 `/usr/bin/python3 -c ...`，`pgrep -fa "^python3"` **匹配不到**；`pgrep -f "nanovllm...api_server"` 又会匹配到发起命令的 bash 自身（曾把执行 kill 的 ssh shell 一起杀掉）。正确写法：`pgrep -fa "^/usr/bin/python3|^python3|^[s]pawn_main"`，kill 前先列 PID。
### I14 🟢 BISECT-1/2 补丁去留未决
- **现状**：`model_runner.py` 的 `# BISECT-1`（replay 前后 rank barrier）与 `# BISECT-2`（每桶私有 graph pool）。8B 当时是实证修复（5/5），本轮隔离下 8B 仍 0/5 → **该修复不构成通用解**。
- **下一步**：I1 有结论后一并处理——图若要重开则保留并重验；图永久放弃则删除，降低认知负担。当前对 eager 路径无影响，可暂留。

---

### I19 🔴 每步固定成本 ~1.4–2.2 ms/层（当前最大性能杠杆）
- **实测（10-27）**：0.6B 与 1.7B 的 b1 步时**都是 ~63.8 ms**（若带宽 bound，1.7B 应慢 2.5–3×）；TP1 只比 TP2 快 ~12 ms（all-reduce 不是主因）；0.6B eager b1 的 63.2 ms 中 **forward 占 98.2%**（62.1 ms），prepare 0.22 ms、sample 0.86 ms。
- **推论**：0.6B 的权重带宽下限只有 ~3 ms/步，实测 ~40 ms 净 GPU 时间 → **约 90% 是每层固定开销**（28 层 → ~1.4 ms/层；8B 32 层 → ~2.2 ms/层，P6b 74.3 ms 亦吻合）。
- **为什么它最优先**：同时压着 (a) 单流延迟（30B b1 = 165 ms/step）、(b) 容量（0.94 req/s 硬饱和）、(c) 所有"多步"方案（投机解码需要 k+1 个 draft 步；这正是 S1 判 no-go 的技术原因）。收益面大于投机解码，且不依赖产品决策。
- **30B 分解已完成（10-27）**：b1 步 177.6 ms 中 **forward 169.6 ms（95.5%）**、prepare 0.27 ms（0.15%）、sample 7.7 ms（4.3%）；b8 96.0%/0.15%/3.9%；**b32 的 sample 升到 36.6 ms（13.7%）**。→ 固定成本在 **forward 内部（层内 kernel）**，不是 Python/调度/采样路径；b32 的 sample 是独立的便宜目标（`bool(greedy.all())` 同步 + argmax）。
- **逐层计时已完成（10-27，两次重复）**：0.6B TP1 eager b1 → **launch/CPU-bound，不是单 kernel 慢**。步时 53.7–61.2 ms，kernel 设备时间 29.3–32.4 ms → `device_busy = 0.478–0.603`（门 <0.70）；每步 **2,401 次 launch**（35 种 kernel，平均 **13.5 µs**/launch）；每层 host 发射 1.84–2.13 ms × 28 ≈ 整步步时 → **CPU 发射是关键路径，GPU 空转约 40%**。
- **成本构成**：top-15 kernel 占设备时间 74%，几乎全是 RMSNorm/RoPE/mask 的逐元素小 kernel（Cast fp32↔bf16、Mul/Add/Rsqrt/Pow/Select/Concat）；真正的 flash attention 只有 28 次 / 2.3 ms，GEMV/GEMM 合计 ~3.4 ms。**算力不是瓶颈，是"小 kernel 洪水"。**
- **下一步（按 ROI）**：① **纯 eager 的 norm/rope 融合**（消掉每层几十次 launch，不依赖图，可逐项 A/B）——新发现的可执行方向；② 图路径（I1/I2）仍是 63.2→40.0 ms 的现成 23 ms，但被 replay 漂移卡住；③ 30B TP2 抽查一次，确认 launch-bound 在 MCCL 下仍成立。
- **证据**：`benchmarks/P6-layer-timing.md`、`benchmarks/data/p6d/p6_layer_timing_06.json`（sha256 `ec7e02078d8bb9695737f91c5bb7f9fa8739a4259ff7119f740806db9377eef3`）、`/root/p7_step_overhead_06.json`、`benchmarks/P7-phase0.md` §2

### P8 规划新增(2026-10-28 执行轮更新;结论标注在每条)

> 来源文档(P8-architecture-review 等)见 git 历史;P8/P9 证据见 `benchmarks/archive/` 与 `benchmarks/archive/P9-*.md`。

- **I20 KV 量化(🟡 P19 判决:per-channel 机制逐位正确;通道内动态范围 500× 超出 int8 承载 → 待混合精度 outlier 通道)**:P17(隔夜轮)实现全链路 opt-in `kv_cache_dtype="int8"`:池 [2,L,blocks+1,256,kvh] int8 + scale [2,L,blocks+1,256,kvh] fp32;融合量化写 kernel(`nanovllm/kernels/kv_int8.py`,对 torch 参考逐位);paged decode kernel 融合 dequant(HAS_SCALE 分支)+ `_gather_kv`/sdpa 回退同。**容量 2.0× 精确(730→1460 块)、decode 步时 1.013×(≤5% 门过)**。**数值门 FAIL(照录)**:0.6B e2e 生成退化(重复/乱码),32/32 请求首 token 分叉。**根因(实测)**:Qwen3 K 激活 per-head amax 196~492 而典型元素 ~0.5-0.8 → per-token-head absmax 量化步长 1.5-3.9 淹没信号(P14 探针 0.67% 系无离群点合成数据,误导)。逐件排除 plumbing:store/gather 逐位、paged+scale vs torch 重算 4.8e-4 ≈ bf16 基线、引擎内 decode 注意力现场重算 diff 0.0019。**P19 修复尝试(per-channel 校准静态 scale)**:已实现并逐位验证(store BITWISE、引擎内 manual-quant 全行恒等、bulk 量化级数中位 60/127);**质量门仍 FAIL — 根因再升级:通道内动态范围 ~500×(channel amax 398~492 vs 典型 ~1),int8 的 127 级绝对网格数学上无法兼顾;服务分布超校准 amax 时裁剪误差实测 worst 27.5**。**fp8 e4m3 on MUSA = 静默 no-op**(cast 不量化,492 零误差=假实现,比崩溃更危险;厂商工单素材)。**最终方案(下轮立项)**:混合精度 outlier 通道(高动态比通道保 bf16,其余 int8,~1.03× 存储;读侧双源)——布局与读路径全部就绪。默认 bf16 不变(预注册)。历史(P14):朴素 dequant pass 慢 bf16 拷贝 5.7× → 必须融 kernel(已实现)。证据 `benchmarks/data/p14/int8_kv_probe.json`、`/root/p17_data/gate.json`。
- **I35 stall_free decode 循环 for-range 越界(🆕 P16 发现并修复)**:`_schedule_stall_free` 的 decode 循环按 `num_running` 固定迭代,但每次抢占额外从 running 弹出 victim → 抢占多发时 popleft 空 deque IndexError(上游潜伏,生产抢占罕见未触发);P16-C 压力负载触发,修为 `while self.running`。
- **I21 per-rank KV 容量不一致(✅ 本轮修复)**:`allocate_kv_cache` 后 `dist.all_reduce(MIN)` 统一两 rank 的 `num_kvcache_blocks`;tiny TP2 六配置回归全等 rc=0,两 rank `[kvcache] blocks=` 一致(343080/343080、341994/341994;本轮加载内启发式恰好相同,真 divergence 钳制待 30B 长上下文确认)。
- **I22 graph replay padding 写 KV slot 0(✅ 根因确认 + 修复 + TP1 验证;TP2 另有独立故障)**:scratch-slot 修复(`allocate_kv_cache` 多留 1 块 + replay 前 `slot_mapping/block_tables` 填 scratch);tiny/0.6B/8B 验证:GB 5/5 token 全等(paged)、GC 共享前缀全等、GD 捕获零失败;**30B TP2 首次 replay 楔死是独立故障**(见 I29),30B 图维持否决、默认 eager。详见 `benchmarks/archive/P8-graph.md`。
- **I23 mixed step 走 prefill MoE 后端(✅ 本轮修复,效果待 L3 在线复测)**:`model_runner.run` 对 mixed 拆 prefill/decode 子批,decode 行走 `prepare_decode`+paged+bmm_dense(`NANOVLLM_MIXED_SPLIT`,默认开);调度器 mixed-batch 契约有单测(`tests/p8_scheduler_gaps.py`)。
- **I24 host sync 浪费(🟡→✅ 部分,本轮实测)**:prefill 96 次 `.tolist()` 上提到 prepare(实测 0.6B 512-token prefill 步时 **−11.6%** / 62.8→55.5 ms;30B 复核在 L2)+ sampler 2 次 `bool()` 改 host 侧 `all_greedy`(0.6B b32-mixed sample 段 −13.5%,30B L2 复核)。**重大副发现:launch −960/步 + device −10.7 ms 但步时仅 −0.8%(fused RMSNorm A/B)→ decode 步时对 launch 数与 host sync 均不敏感,I19 的"launch-bound→融合换步时"因果链被实测否定**;每步仍有 ~20–30 ms 未解释的固定等待,整图 replay 被 TP2 wedge 阻断(见 I29);分段图(图内零 MCCL)已于 P11 过 0.6B bitwise+异步门,步时判决实验随 P11 P3/30B 双轨进行。
- **I25 benchmark v3(✅ 本轮落地)**:`benchmarking.py` 增 wilson_ci/token_goodput/classify_error/conforming/linear percentile;客户端 schema v3(双 SLO、error 分桶、timeout→服务端 cancel、--repeats+CI、--mode concurrency、mixed 3:4:3、GPU 采样、p99_reference_only);服务端 final-chunk 带 metrics;单测 15 绿 + v2/v4 JSON 回放不抛异常。
- **I26 KV admission/backpressure(✅ 本轮修复,端到端待 L3)**:`add_request` 校验 `prompt+max_tokens > max_model_len → 400`、`> KV 容量 → 429 + requests_rejected 计数`;scheduler 容量不足返空批次(mode="empty")不 assert;HOL `can_allocate==-1` 改跳过候选继续尝试 + `kv_admission_skips_total` 计数(FIFO 序保持);单测 6/6(`tests/p8_scheduler_gaps.py`)。
- **I27 sdpa-decode 图的烘焙形状数值限制(🆕 本轮定位)**:capture 时 context_lens=256 固化 kv 窗口形状,SDPA 归约切分随 shape 变 → 1-ulp 级漂移与近平局翻转(tiny MoE graph 5/5 自洽但不等于 eager);paged kernel 循环边界在 device 上,不受影响 → 生产路径(paged)graph==eager bitwise。sdpa 模型要用图需按运行时窗口分桶捕获。
- **I28 torch.musa 跨块步长视图 D2H 拷贝返回不一致数据(🆕 平台 bug 嫌疑)**:`kv_cache[:, :, blk, 0]`([2,L,kh,d] strided view)的 `.cpu()` 返回 NaN 垃圾,同位置 K/V 分读与 device 侧逐层扫描干净,`.contiguous()` 后仍复现;workaround = contiguous 化或分 K/V 读;影响所有"整读跨块切片"的诊断代码。
- **I29 TP2 引擎级图 replay 楔死(P9 定性收口,10-28;P14 追加线索见末尾)**:**间歇性 rendezvous 楔死,非结构确定性**。P9 分级探针(`tests/p9_mccl_graph_probe.py` + `p8_graph_padfix --tp 2`)实测:① 纯张量图(96 图内 AR、b32 payload、48×bmm+AR、~2400 节点、**六桶共享 pool 完整复刻引擎 capture/replay 协议**)TP2 全绿零楔死;② TP1 引擎级从未楔死(P8 0.6B/8B 5/5);③ **0.6B TP2 paged 首次 replay 即楔死(3/3 复现,最小复现,2 min 加载)**;④ 0.6B TP2 sdpa 撞过 2 次 generate 后楔死 → paged 提高首中概率但非必要;⑤ **replay 前后 barrier(`NANOVLLM_GRAPH_REPLAY_BARRIER=1`,默认关)不是解**。必要成分 = 引擎 replay 路径(graph_vars 每步 fill/scratch 填充/replay 后贴 eager compute_logits/采样节奏)× TP2 collectives。崩溃形态多变(对称自旋/单 rank 死+另一 rank 旋),均需按 PID kill(I12)。**30B 图双轨门维持结构性阻断,默认 eager 不变**。复活条件:分段图(attention eager + linear/MoE 进图)或厂商 MUSAGraph+MCCL 修复后用本探针回归。证据:`benchmarks/archive/P9-graph.md`、`benchmarks/data/p9/p9_mccl_graph_probe*.json`、`p9_06b_padfix.log`。**P10 进展(分段图原型,`NANOVLLM_SEGMENT_GRAPH=1`,默认关)**:每层两段(AR 全部移出图)的 segment graph 在 TP2 **捕获可行**(tiny 16 图双 rank)且 **replay 首次稳定跑通**(3×3×8 token 零楔死、自洽、token 与整图一致;`tests/p10_segment_smoke.py`)——但全程 LAUNCH_BLOCKING 下;异步捕获仍撞 I3 族偶发与一次 wedge,padfix 双引擎上下文捕获 illegal memory access 未解;0.6B paged GB 门未跑。见 `benchmarks/archive/P10-graph.md`。**P11 进展(两进程口径 + S2/S4 双过,分段路线转已验证)**:padfix 加 `--phase eager|graph` 两进程口径绕开双引擎上下文后,**0.6B TP2 paged S2 主判决 PASS**(224 图/rank 零失败、GB 5/5 bitwise、GC、GA、零楔死,LB 下)+ **异步门 PASS**(无 LB 捕获首攻成功 + GB 5/5 bitwise——分段图在生产模式数值正确且稳定)。P10 的异步偶发与双引擎 illegal access 本轮 3/3 全部未复现(含 SEG_DEBUG 打点版)——illegal access 降级为偶发未复现、打点开关保留。**P11 P4 收口(全梯子验证完成)**:8B 门 PASS(288 图零失败、GB 5/5 bitwise、异步零楔死)+ **30B 双轨门五项全 PASS**(768 图/rank 零失败、跨臂逐位 0 翻转 b1/b8/b32、显存 +0.00 GiB、b1 decode **−23.7%**(152.85→116.63 ms)超 ≥+15% 转正门;b8 持平、b32 +3.8% 真实回退)。**I19 固定等待判决:b1 的 ~36 ms/步确为逐算子 host launch 开销,replay 可回收**。**P12 终局:L3 在线 A/B 判定 FAIL**——四档 TPOT p95 比 eager 慢 5.9–14.7%(门②③均不过),B2 同 session 复跑把 B' 复现到 0.1% 内;归因见 I33。**分段图不转正,默认 eager 不变,仅单流场景可显式开**。见 `benchmarks/archive/P11-graph.md`、`benchmarks/archive/P12-serving.md` §P0 判定。**P14 追加（联网线索 + 实测）**：摩尔线程 [MCCL×MUSA Graph 文档](https://docs.mthreads.com/musa-sdk/version-5.2.0/libraries/mccl/mccl_user_guide_2.4/usage/musagraph.zh-CN/) 明言「被捕获与未捕获的 MCCL 操作混用时，多通信器从同一线程 `musaGraphLaunch` 已观察到死锁」，开关 `MCCL_GRAPH_MIXING_SUPPORT`（默认 0）。我们的引擎正是「图内 AR + replay 后 eager AR」混用，仓库此前从未设过任何 `MCCL_*`。0.6B TP2 paged 整图最小复现，交错 ×(3+5) 实测：base **0/8**、`=0` **0/3**、`=1` **3/9** 完整通过（GA/GB/GC 全 PASS、与 eager 逐位一致，单 run 36 s），Fisher 单侧 p≈0.07。**判定：缓解非修复**——混用是成分之一但不是唯一楔死源；30B 整图红线不放开、默认不改。详见 `benchmarks/archive/P14-mccl-mixing.md`。未排除：8B TP2 复测、`MCCL_GRAPH_HELPER_DISABLE`。
### I30 ✅ 已关闭（P13 ABA 定案，10-30）：「现行栈回归」主张被否定，方向反转

**终局（同 session ABA：旧树 d5779b6 → 现树 → 旧树，三加载）**：现树两档 TPOT p95（0.9:0.2612、1.0:0.2894）**全部 12 个 repeat 配对快于旧树 23–51%**（旧树 1.0 档 0.580/0.588，slo_standard 仅 ~75%；现树 6/6 全 100% 双口径）。判定脚本标 INVALID 仅因旧臂自身 CV 8.5–12.1%（旧栈慢且噪），关闭条件「当前两档均≤两旧臂」逐字满足且余量压倒。**P9 的 0.387–0.397 与 v4 单跑 0.327 归入机时状态/单跑口径伪象**（与 P12 臂 A′ 0.2691 互证）。v4 门 TPOT p95 ≤0.343 保持为生产回归门（现树 0.26–0.29 达标）。证据：`benchmarks/archive/P13-i30.md`、`benchmarks/data/p13/aba_*.json`。历史记录：P9 发现 0.87 档 0.387–0.397 超 v4 门 0.343 → P10 排除 mixed-split 税 → P11 复核中止 → P13 ABA 定案关闭。

### I31 🟡 30B 进程首个 batch 与稳态逐位不同（一次性 warmup 态跃迁，P11 发现）

30B TP2 生产栈 greedy decode,进程内第一个 batch 的 token 与之后所有 batch 不同(run1–4 两两逐位相同,跨 512-token chat prompts、b1/b8、64 decode;`tests/archive/p11_30b_flip.py` 证据 `benchmarks/data/p11/p11_30b_flip_eager.json`)。**稳态 greedy 是逐位确定的**(MoE 原子加非确定性嫌疑排除)。含义:① 所有 token 一致性门必须**丢弃首 batch**(或先跑一个 warmup batch 再比对),否则把 warmup 态跃迁误判为数值漂移(本轮 p11_30b_ab 的 tokens_consistent=False 即此假象);② 机制未查(候选:首 batch 触发 JIT/autotune/allocator 布局一次性切换)。跨臂(eager vs segment)稳态逐位 0 翻转 = 分段图数值等价的最终证据。

### I34 🟡 引擎 token 依赖状态历史：处女引擎与跑过流量的引擎在同 prompt 上近平局分叉（P15 发现，泛化 I31）

0.6B TP1 paged eager，同引擎内先跑 6 个 PD 交接请求再跑 collocated 生成，与处女引擎的 collocated 结果比：**2/6 请求于中途近平局分叉（completion#10/#19）**，且每种状态历史下各自跨进程逐位确定（PD×2、BASE×2 全等）。含义：① **bitwise 对照臂必须同引擎同状态历史**——处女 baseline 不是合法参照（P15 M2 门已改为同引擎对照，`scripts/p15_pd_1p1d.py`）；② I31 的"首 batch 跃迁"实为"状态历史相关"的特例，机制候选同 I31（allocator 布局/JIT/autotune 一次性切换）；③ 对所有 token 门的影响与 I31 ①相同，但作用域从"首 batch"扩大到"任何状态历史差异"。证据：`/root/p15_data/pd_m2_verdict_*.json`（run1/run2 + in-engine collocated），报告 `benchmarks/P15-pd-autotune.md` §E。

### I32 🟡 api_server「默认启动」≠ 生产栈；30B 段图捕获的显存前提（P12 发现，两处）

① `--decode-moe-backend` 默认 None = 跟随 `--moe-backend`(grouped_mm),**不带 flag 的 api_server 启动 decode 跑在 grouped_mm 上**——不是文档生产栈(bmm_dense decode;P6-serving-v3/v4、P9-serving 均显式带 flag)。首轮臂 A/B 因此整体作废,臂 B 捕获嵌入 grouped_mm 直接 `operation failed during capture`。② 30B 段图捕获桶集 = `min(max_num_seqs,512)` → 512 默认 = 35 桶 3360 图,且 KV 按 0.9 包络先填满 → **1632/3360 图处 OOM**(图池 5.91 GiB 时);P11 验证形态是 max_num_seqs=64 → 8 桶 768 图、池零增量、捕获 5.2 s。**结论:段图上服务必须 `--max-num-seqs ≤ ~64`(本矩阵实测最大并发 37,饱和段 bs>64 步自动回退 eager,无失真)或先做 KV 预留改造**;转正落地的前提条件(见 P12-serving.md)。api_server 已补 `--decode-moe-backend` 之外的 `--enforce-eager/--no-enforce-eager` 与 `--max-num-seqs` 参数(默认值不动,生产栈不变)。

### I33 🟡 桶化图的 padding 税;离线图门只测桶边界(P12 发现,已致分段图在线否决)

- **现象**:分段图离线 b1 **−23.7%**,在线四档 TPOT p95 反而**慢 5.9–14.7%**(A' 0.1968/0.2519/0.2372/0.2691 vs B 0.2108/0.2743/0.2707/0.2848);同 session 的 B2 复跑把 B' 复现到 0.1% 内,排除机时污染。
- **机制**:捕获与 replay 按桶(`model_runner.py:709` → `[1,2,4,8,16,32,48,64]`),`cap = next(x for x in graph_bs if x >= bs)`(`:835`)**整桶执行**;在线矩阵实测 max_concurrency 14–30 落在桶之间,batch 14 按 16 跑、21–27 按 32 跑。被 pad 的行只把 `slot_mapping`/`block_tables` 指向 scratch 块(数值无害,I22),但 attention/MoE 的**计算是真的**。
- **为什么门没抓到**:P11 的 30B 双轨门只测 `b1/b8/b32`(`tests/archive/p11_30b_ab.py:61`),恰好全是桶边界,pad 比 = 1.0 → b9–b31 是结构性空白。
- **与既有观测自洽**:b8 起进入设备饱和区(host launch 收益被 eager AR 与设备计算抵平),b32 已是 +3.8% 回退;在线并发正落在这条带上。
- **处置**:段图维持默认关(仅单流可显式开)。若要在多并发下复活:① 12–32 间加密桶(步长 4,图数 768→~1200,先核图池显存);② `cap/bs > 1.15` 自动回退 eager;③ **离线门必须补 b12/b20/b24**。
- **普适教训**:任何"按桶捕获"的路径,验收门里必须有**非桶边界**的 batch 点,否则 padding 税必然漏。
- **证据**:`benchmarks/archive/P12-serving.md` §P0 判定、`benchmarks/data/p12/p12_online_arm{A2,B2}.json`、`p12_ab_verdict{,_samesession}.json`。
- **P13 终局(10-30,✅ 闭合:修复已实现并验证,分支定格关闭)**:①0.6B 探针判定力不足(阶跃仅 +3.3%,设备计算太小);②**修复实现**——`NANOVLLM_SEGMENT_GRAPH_BUCKETS=dense`(vLLM 式 11 桶,30B 1056 图/rank 零失败 37.06 GiB)+ `NANOVLLM_SEGMENT_GRAPH_PAD_RATIO=0.15`(cap/bs>1.15 该步回退 eager),**默认惰性**;0.6B paged bitwise 门 10/10 全等、回退精确触发;③**30B 同 session A/B**:padding 税被证实并移除(0.9 档 +14.7%→+2.3%,0.5 档 +7.3%→+1.0%),四档 SLO 全 100%,但 1.0 档残余 +4.7% 超 F2 门(+2%)、F3 红利门 FAIL → **转正失败,分支关闭**;残余 +1.0~4.7% 恰等于 P11 桶边界固有开销(b8+0.5%/b32+3.8%)= **每层 2 次 eager AR 串行化**,属 I29 范畴(AR 进图需整图,已否决)。**分段图定格为单流可选开关;该服务化路线不再投入。**证据:`benchmarks/archive/P13-{padding,fix}.md`、`benchmarks/data/p13/fix_*`。


> 详细论证见 git 历史 `docs/P7-speculative-decoding-survey.md`(必要性、选型、判决表)与 `docs/P7-implementation-plan.md`(代码方案、算法、测试门);判决报告见 `benchmarks/archive/P7-*.md`。

### S1 ✅ 已判决 NO-GO（10-27 Phase 0 实测）：朴素 draft 路线不成立
- **实测**（`benchmarks/archive/P7-phase0.md`）：α(0.6B→30B) = **0.0056**、α(1.7B→30B) = **0.0068**（32 条 prompt，LCP 均值 0.72/0.88，最大 4/7，零条完全一致）；c = **0.384/0.386**。最优 k=1 的加速 **0.725×/0.741×** —— **比基线慢 26–28%**，与冻结门 1.3× 相差 44%。
- **两个独立否决原因**：① draft 太慢（c≈0.39，源于每步固定成本，见 I19）；② 接受率≈0（0.6B/1.7B 是 Qwen3 初代思考型后训练，target 是 2507 非思考；跨代组合导致首 token 分布即分叉）。
- **已排除的伪因**：tokenizer 三模型逐 token 一致（14/14/14）；引擎收 raw encode 无模板差异；草稿输出连贯非退化（48 token 34 个不同，与 target 前 2 token 相同）。
- **决策**：git 历史 `P7-implementation-plan.md` 的 M1–M4 **不启动**；S6（EAGLE3/PARD 训练头）成为唯一可能路线，且**前置条件是先解决 I19**（否则 α=0.8 也只有 ~1.1×）。
- **证据**：`/root/p7_target_ref.json`、`/root/p7_draft_steps.jsonl`、`/root/p7_alpha_06_tp2.json`、`_17_tp2.json`、`/root/p7_step_overhead_06.json`

### S2 🟡 M1：双模型加载 + 多位置 logits（Phase 0 通过后开工）
- **改动**：`embed_head.py:58-60` 只回 last-token logits → verify 需多位置；`ModelRunner` 需第二套 draft config/model/kv_cache（当前严格单模型）；Config 增 3 字段（`draft_model`/`spec_k`/`spec_enable`，否则被 `llm_engine.py:34-37` 过滤掉）。
- **风险**：`get_rope` 是 `lru_cache(1)`，两模型共存需确认参数键；`torch.set_default_dtype/device` 切换舞需正确重入。
- **门**：T1（0.6B 自草稿自验证，α 应=1.0）+ logits 形状冒烟。

### S3 🟡 M2：多 token 提交 + 回滚 + 前缀缓存脏块
- **改动**：`scheduler.py:229` postprocess 硬编码 1 token/seq → `postprocess_spec`；verify 前需保证每 seq 剩余容量 ≥ k+1；**块边界回滚**需 `release_last_block` + prefix-cache hash 作废（否则未来序列会命中含被拒 token 的脏块）。块内回滚零改动（下个 decode 步重算同一 slot）。
- **MVP 绕法**：检测到拒绝点恰在块边界时 `clear_prefix_cache()`（正确但伤命中率，概率 ~1/256/轮）。
- **门**：T2 **逐 token 全等**（32 条 chat prompt × 128 token，vs `p7_target_ref.json`）——这是 spec 的代数保证，硬门。

### S4 🟡 M3：draft KV cache + 批化
- **改动**：draft 也要 paged KV（草稿步 O(1)）；batch 内每条 seq 接受数不同 → 需要 per-seq 状态机。
- **风险**：draft TP2 的 MCCL 开销（28 层×2×0.08 ms≈4.5 ms 只是下限）；若 all-reduce 占 draft 步 >20%，切换**双 rank 整卡副本各自贪心**（确定性 → 零同步，代价 ~0.6 GiB×2）。
- **门**：T3 单流 TPOT 兑现 Phase 0 预测的 ≥90%。

### S5 🟢 M4：温度拒绝采样 + API 接入 + 在线复测
- **改动**：标准拒绝采样（`min(1, q/p)` + `(q−p)_+` 修正分布）；draft 采样时保留 `p_i(t_i)` 标量；`api_server` 加 spec 透传；在线四档复测（v3 口径）。
- **门**：T4 分布一致性（KL 对比基线波动）；T5 100% 档 SLO 不劣化且 goodput 上升。

### S6 🟢 EAGLE3 / PARD 训练头（远期）
- **内容**：α 可到 0.8+（判决表从 ~1.55× 升到 ~2.3×），但需：target 返回倒数第二层 hidden、EAGLE 树验证、权重接口适配。社区已有 Qwen3-30B-A3B 的 EAGLE3 权重，可直接下。
- **触发条件**：S1–S5 全部落地且产品需要进一步降延迟。

---

## 3. 生态与厂商（E 系列）

> 核心结论：**能手搓两处**（E1 我们的 paged kernel、E2 vLLM fused_moe 配置），**搓不动四类**（E3 grouped_mm、E4 MUSAGraph、E5 mp22、E6 watchdog）——后者是 driver/runtime 闭源层。

### E1 🟢（已解闸，低优先）手搓 paged decode kernel 的 autotune
- **可做性**：✅ 单点注入。`nanovllm/kernels/paged_attention.py:128-130` 写死 `BLOCK_N=32, num_warps=4, num_stages=1`；全仓库无 `@triton.autotune`。
- **方案**：扫 `BLOCK_N∈{16,32,64,128} × warps∈{2,4,8} × stages∈{1..4}`（≤64 组）× 桶 `(batch∈{1,4,16,32,64}, kv∈{≤1k,≤4k,>4k})`；计时复用 `scripts/p6_moe_microbench.py` 的 `time_call` + 子进程隔离；产出 `configs/paged_decode_MTT_S4000.json` + sha256 绑定 kernel 源。预算：一发 30B 加载 30–60 min。
- **硬约束**：`context_len` 必须保持 runtime 循环边界（不然破坏 kernel 复用）；**autotune 只改性能，不改变 SDPA↔paged 的 bf16 数值路径差异（I4）**——不要指望 autotune 解决 I4。
- **触发条件**：**已满足**——I4 已于 10-27 转正（L1–L5 全过，paged 成为默认）。现在可做，但与 I19 的 norm/rope 融合相比 ROI 更低，排在 P8 计划之后。

### E2 🟢 手搓 vLLM fused_moe 的 S4000 配置（原 I16 重新归类）
- **纠正**：此前记为"⚪ 厂商侧不可修"——**不准确**。缺的是 `configs/E=64,N=768,device_name=MTT_S4000.json` 这个**文件**，而生成它是我们的工作。
- **两条搓法**：A（零代码）`VLLM_TUNED_CONFIG_FOLDER` 指向自生成 JSON，扫 `BLOCK_M/N/K × GROUP_M × warps × stages` 按 M 桶取最优（git 历史 `docs/P5-moe-optimization-survey.md:64` 已规划）；B（进程内）走 `nanovllm/kernels/vendor_moe.py:29-45` 的 source-rebind 注入 config dict。
- **低优先理由**：decode 侧 bmm_dense（1.27–2.86 ms/层）已快于厂商 kernel 调好后的水平；只有 autotune 后的厂商 kernel 在 **prefill 大批（2k–4k）** 明显快于 grouped_mm 时才值得（用 `p6_moe_microbench.py --batches 512,2048` 判）。
- **附带小项**：`nanovllm/kernels/moe_fusion.py:61-64,80-85` 两个手写 fusion 的 block/warps 也有微调面，价值更低。

### E3 ⚪ `torch._grouped_mm` 无任何用户可见旋钮
- **现状**：`qwen3_moe.py:248,256` 是唯一调用点，固定厂商 op，约 1 TFLOP/s 固定开销上限（`:284-285` 注释）。
- **处理**：**绕过而非调优**——decode 已用 bmm_dense；prefill 保持 grouped_mm。

### E4 ⚪ MUSAGraph 的规模/依赖链限制（I1 的厂商侧）
- **P14 更新**：① 「~2048 图可执行上限」已被 [SGLang PR #27537](https://github.com/sgl-project/sglang/pull/27537) 与 torchada 文档交叉确认为厂商已知驱动限制；② MCCL 官方文档点名图内/图外 MCCL 混用死锁，`MCCL_GRAPH_MIXING_SUPPORT=1` 在 0.6B TP2 上把完整通过从 0/11 提到 3/9（见 I29）。仍需厂商：剩余楔死源的诊断。
- **现状**：控制组 0/5、无单一代码根因；driver 层不可修。
- **要问厂商**：MUSAGraph replay 在"48 层 × 96 次图内 all-reduce + 图内 bmm"规模下的已知限制；是否有 graph-level debug/dump；以及**为什么摩尔自家 vLLM 分叉在 S4000 上默认对 MoE 开 MUSA Graph**（git 历史 `docs/P6-musa-graph-survey.md`；核心事实已并入 I29）。
- **我们侧**：`enforce_eager=True` 保持；不猜、不试。

### E5 ⚪ 厂商 mp22 fast path 数值错误（原 I17）
- **现状**：rel 1.3–341、b64 NaN，快 7–15× 但永不可用。
- **处理**：报厂商（附 `/root/p6_moe_microbench2.json`）；代码 `use_mp22` 默认 False 保持。

### E6 ⚪ MUSA launch watchdog 诊断能力弱（原 I18）
- **现状**：设备楔死后 10 分钟才杀，报错点漂移（replay 楔死、barrier 报错）。
- **处理**：向厂商确认诊断手段；我们侧已用 `p5_graph_ab.py --verify-watchdog-s` 自救（I11）。

---

## 4. 已关闭（8 项）

| ID | 结论 | 证据 / commit |
|---|---|---|
| I5 | 采用栈 b64 = **141.48 tok/s**（CV 0.0028，3/3 无崩溃，token 三轮一致；vs P5 默认 111.75 **+26.6%**） | `/root/p6_i5_i6.json`，`benchmarks/P6-i5-i6-i7.md` §1 |
| I6 | 前缀缓存 hot 使 TTFT p95 **1.256→0.934 s（-25.6%）**，吞吐仅 +1.5%（价值在首字延迟） | 同上 §2 |
| I7 | 拐点**不是锐利点**：同一目标负载两次跑出 100% vs 84.4% → **高方差软边界 80–100%**，硬饱和在 ≥1.09 目标 QPS（0.94 req/s / 60 tok/s） | `/root/p6_i7_knee.json`，同上 §3 |
| I9 | TPOT 非单调已解释：由瞬时并发决定（29→0.414 s，45→0.593 s），并发由 Poisson 到达尖峰决定 | 同上 §3 |
| I10 | 回归脚本 rc 语义已修：graph 段改 `--with-graph` opt-in，六配置全等门单独决定 rc | commit `91e3fd3` |
| I11 | graph A/B verify 加硬 watchdog（`--verify-watchdog-s`，超时打印卡点并 `os._exit(97)`） | commit `91e3fd3` |
| I13 | `benchmark_online.py` fraction 断言放宽为 `>0`（乘数语义，可 >1） | commit `91e3fd3` |
| I15 | 本地 ssh 别名 `s4000b` 端口 18484→28036 且改用已装的 `id_ed25519` | commit `91e3fd3` |

---

## 5. 优先级总表（10-27 Phase 0 后重排）

| 优先 | 条目 | 理由 | 成本 |
|---:|---|---|---|
| 1 | **I19 每步固定成本** | 新测量的最大杠杆：0.6B 90% 的步时是每层固定开销；同时压单流延迟、容量、以及所有多步方案 | 先 2 个 ≤15 min 的分解测量 |
| 2 | **I8 重设计重测** | 长上下文覆盖仍是空白（本轮无效跑）；需要 token 长度控制 + rank 存活监控 | 1 发加载 |
| 3 | **I12** teardown 超时 | 本轮又踩一次（rank1 卡 barrier）；每轮都要人工 kill | 1–2 h |
| 4 | **I1/图** | 若 I19 确认 launch-bound，图的 1.55×（0.6B b1）就变成高优——但需先解决 replay 漂移 | 时间盒 2–3 h |
| 5 | **E2** fused_moe 配置 | 便宜，但只在 prefill 大批需要时才有价值 | 30–45 min |
| 6 | **E1** paged autotune | 仍被 I4 产品决策闸住 | 30–60 min |
| — | ~~S1 投机解码~~ | **已判 no-go（0.73×）**；S2–S5 不启动 | — |
| — | S6 EAGLE3/PARD | 唯一可能的说明路线，前置是 I19 | 周级 |
| — | I2/I3/I4/E3–E6 | 等厂商证据或产品决策 | — |

**关键依赖**：S1 → {S2, S3, S4} → S5 → S6；**I4 产品决策 → E1**；I1 根因 → {I2, I14}。A（I8）与 B（S1）无依赖，但**都抢同一块 GPU，必须串行**。
