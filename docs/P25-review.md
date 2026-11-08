# P25 技术评审 — 竞品比对 × 代码审查(2026-11-08)

> 本文回答三个问题:① 对标 2026 年主流推理引擎,我们到哪了、差在哪;② 竞品比对给出的改进优先级;③ 本轮深度代码审查发现了什么(含两个**已复现**的 P23 缺陷)。
> 性质:评估 + 提案。§4 的门都是**草案**,正式立项进 PLAN.md 时按惯例预注册冻结;§3 的发现建议回填 ISSUES.md(I37/I38)后修复。数据来源见文内标注;竞品数字为量级(2026-11 检索)。

---

## 1. 三十秒结论

- **成色**:功能面已覆盖 2026 年服务端引擎"标配清单"的绝大部分,相当于**单人复刻了 2024 代 vLLM/SGLang 的核心架构,并追加了部分 2025 代调度器特性**(overlap 调度、优先级/SLO、PD 分离、结构化输出),再叠加一个主流引擎没有的维度——MUSA 非标平台全栈踩坑。在同类个人引擎项目里,这已是**稀缺档位**;短板不在功能清单,而在"性能杠杆矩阵"(量化/投机/graph)与性能杠杆矩阵之外的工程叙事。
- **本轮审查净产出**:2 个已复现缺陷(均 P23 overlap 路径,opt-in,生产默认不受影响)+ 若干低危项;生产主链路(串行 stall_free / 调度 / MoE / 采样 / API)审查未发现新的正确性问题。
- **最优先动作**(成本从低到高):修 F1/F2 + 回归 → overlap 服务端产品化 → 工程讲解文档 → 量化路径立项。

---

## 2. 竞品比对(2026-11)

### 2.1 主流引擎现状

| 引擎 | 版本/活跃度 | 定位 | 与本项目最相关的特征 |
|---|---|---|---|
| vLLM | v0.33(2026-11),月更,~96k stars,Octoverse 活跃度第 2 | 事实标准 | V1 统一调度器 + 原生 overlap;全栈低精度 KV(v0.32);llm-d/Production Stack 生态;Stable ABI |
| SGLang | v0.5.x,数万 stars | 大规模生产/agentic | 零开销调度器;RadixAttention + cache-aware 路由;DeepSeek 级多机 EP;PD 一等公民;XGrammar |
| LMDeploy | v0.18(2026-09),~6-7k stars | TurboMind C++ 引擎 | decode 性能招牌(AWQ/MXFP4 ~1.5× vLLM);投机解码;多模态 |
| TensorRT-LLM | v1.0 GA(2025-09 起) | NVIDIA 官方 | PyTorch 后端转正;PD/投机全家桶;深绑硬件 |
| llama.cpp / MLC-LLM | 滚动发版,~80-100k / ~20k stars | 端侧/编译型 | 定位不同,不构成服务端竞品 |
| KV 生态 | LMCache / Mooncake / DeepInfer(2026-10 DeepSeek 开源) | KV 分离/传输层 | Mooncake 三级 KV 池;DeepInfer 日处理 ~155T token(二手转述,引用需复核) |

### 2.2 能力差异标注(✅ 有 / ❌ 无 / ⚠️ 有但受限,按本项目证据)

| 能力 | 本项目 | vLLM/SGLang | 差异标注 |
|---|---|---|---|
| continuous batching / chunked prefill | ✅ | ✅ | **持平**(stall-free decode-first;chunk-256/1024/2048 可调) |
| paged attention | ✅ 自研 Triton | ✅ FlashInfer/FlashAttn | 持平(后端数量少;内核"正确+局部更快"阶段) |
| prefix cache | ✅ block-hash + LRU 淘汰 + touch-on-hit | ✅(SGLang **radix tree**) | **差距:radix tree**(块内部分命中做不到;engine-vs 文档已列为已知差距) |
| 优先级/SLO 感知调度 | ✅(高优 TTFT 42.7×) | 部分 | **持平偏强**(vLLM 无原生 per-request 优先级) |
| overlap scheduler | ✅ opt-in,bitwise 门 | ✅ V1 原生 | 持平(差异:我们是预注册门验证过的 opt-in) |
| 结构化输出 | ✅ xgrammar(guided_json/regex/response_format) | ✅ | **基本持平**;缺 json_object/guided_choice |
| 采样面 | ✅ top_p/top_k/min_p/三 penalty/logprobs/stop_ids | ✅ | **基本持平**;缺 n>1、beam、seed |
| OpenAI API | ✅ chat+completions,流式/usage/cancel/429 | ✅ 完整 | 差距:tools、多模态输入、embeddings 端点 |
| **PD 分离** | ✅ 单机双卡 1P1D 全链路(0.79×/0.82×) | ✅ 多机 | 方向持平,规模差距(硬件只有 2 卡) |
| **量化推理** | ❌ 运行时(fp8=MUSA 静默 no-op;int8 KV 判负) | ✅ FP8/W8A8/AWQ/GPTQ | **最大功能缺口**;但判负过程本身有叙事价值 |
| **CUDA graph** | ⚠️ eager 生产(I29 驱动楔死);分段图已判(在线 FAIL) | ✅ 默认 | **平台阻断**;完整否决链是加分叙事 |
| 投机解码 | ❌(判负 α≈0) | ✅ EAGLE/MTP | 判负(α 实测≈0,在 MoE 小激活模型上这个负结果可信) |
| 多模态 / LoRA | ❌(刻意不做) | ✅ | 与主线无关,README 已声明 |
| 可观测 | ✅ 请求级 TTFT/TPOT/ITL + Prometheus + 步级 JSONL | ✅ | **持平**;缺 Grafana 面板/trace 导出 |
| 多实例/多节点 | ❌ | ✅ | 硬件只有单机双卡,无法验证(已在 engine-vs 声明) |
| 评测方法论 | ✅ 预注册冻结门 + 负结果留档 + provenance | 各自 bench | **强于主流引擎的公开实践**(这是差异点不是差距) |

**结论**:差距集中在四个"杠杆"——量化、投机、graph、多机——其中两个被平台物理阻断、一个被实验判负且有完整证据链、一个是硬件规模;真正可行动的功能缺口是 **radix 前缀树、KV 卸载/传输(吃 PD 线)、json_object/choice、n>1/seed**。

---

## 3. 代码与指标审查(本轮净产出)

**审查范围**:engine 全链路(scheduler/block_manager/llm_engine/model_runner)、sampler、attention、qwen3_moe、api_server、async_engine、kv_channel/protocol、detokenizer、prometheus_metrics、config/sampling_params/sequence/context,共约 3.3k 行核心;paged/kv_int8/moe_fusion 等 kernel 未逐行审(由六配置 bitwise 门覆盖)。P19 轮已做过 8 文件独立 review,本轮发现全部位于其后的 P22–P24 新代码。

### 已复现缺陷(repro:`tests/p25_review_repro.py`,纯 CPU,xxhash+numpy 即可跑)

> **P25 修复轮(同日)状态:F1/F2/F3/F4/F6 已全部修复并验证——见各条"修复"注记;F7 经复核为误报(撤销,fcfs empty 路径经 `schedule()` 重建时带 `schedule_s`);F5 文档已刷新。修复证据:`tests/p25_review_repro.py` rc=0(F1 门单元验证,本地+服务器)、`tests/p25_overlap_fix.py` t1 中途 disable 排空 bitwise 全等 / t2 翻转窗 ABA bitwise 全等(跨 768 块边界区)/ t3 KV 压力负载完成(F1 引擎级),外加 p23 全套 + 四套标准回归(见 §4.2 R1/R2 状态)。**

- **F1(P1,崩溃)投机预检查逐序列不累计 → free list 耗尽 IndexError**。
  位置:`nanovllm/engine/scheduler.py:236-242`(stall_free)与 `:101-114`(fcfs)预检查逐个调 `can_append_at`;随后 decode 循环逐个 `may_append_at` **累计**分配。当 `Σneed > free` 且每个 `need ≤ free` 时,预检查全部通过(不 defer),分配在第二个序列上跑穿 free 列表 → `_pop_free` 对空 deque `popleft()` 崩溃。串行路径靠抢占兜底不受影响;投机路径按设计永不抢占,这是它唯一的防线。复现:free=1、两条 512-token 序列同跨块边界,预检查 [True,True],第二条 `may_append_at` 抛 IndexError。KV 压力下"free≤k 且 ≥k+1 条边界跨界序列"即可触发,窗口不宽但真实。
  **修法**(一行级):预检查累计 `total_need = Σ _blocks_needed`,`total_need > len(free_block_ids)` 即整批 defer。

- **F2(P1,静默错)运行时关 overlap 丢弃已排投机批 → block_table 不变式破坏 → KV 写偏一块**。
  位置:`nanovllm/engine/llm_engine.py:229-236`(`set_scheduler_overlap(False)` 直接 `_sched_next=None`,注释称"drains naturally",但只 drain 了 `_pending`,没回滚 `_sched_next` 的变更)。投机 schedule 对 inflight 的跨界 decode 行已把 block_table 抬到 `ceil((L+1)/bs)`;批被丢弃后 token 永不追加,串行不变式 `len(table)==ceil(len/bs)` 破坏,`prepare_decode` 用 `block_table[-1]` 算写槽 → 复现中 KV 写到 slot 767(应为 511),attention 却从旧块读 → **该序列从跨界点起持续产出坏 token**。同族:`NANOVLLM_OVERLAP_NO_SPEC=1`(E5 诊断环境,llm_engine.py:301-304)每步都排了再丢——诊断时若序列跨块边界同样致污(P23 测试序列偏短,大概率没跨 256 边界,所以 bitwise 门没拦到)。
  **修法**:disable 时先 drain-launch 该批(再正常 collect),或回滚 `may_append_at` 分配(逐行弹回多余块)。

### 低危/一致性项(F3–F7,照录)

- **F3(P2)**:fcfs decode 循环不受 `max_num_batched_tokens` 约束(`scheduler.py:147`);默认 `max_num_seqs(512) ≤ max_num_batched_tokens(16384)` 时无害,但用户设 `max_num_batched_tokens < max_num_seqs` 时,fcfs 投机预检查按 token 预算提前 break,decode 循环却继续排 → 未预检查的行直接 `may_append_at`,与 F1 同族。修法:decode 循环加 token 预算,或 config 里断言 `max_num_seqs ≤ max_num_batched_tokens`。
- **F4(P3)**:api_server `int(body.get("max_tokens", 64))` / `float(temperature)` 未捕获 TypeError/ValueError(`api_server.py:128-129,351-352`)→ 畸形输入 500 而非 400(P19 已修 priority,这两处漏了)。
- **F5(P3,文档滞后)**:`docs/engine-vs-vllm-sglang.md` 能力矩阵还停在 P21 前(采样写"仅 greedy/temperature"、结构化输出 ❌、n>1 行为未更新)——P22/P24 已补齐。这份文档正是读者会看的对标页,滞后会造成系统性低估,**建议随下个 commit 一起刷新**。
- **F6(nit)**:`pd/kv_channel.py` 硬编码 `torch.musa.synchronize()`(`utils/compat.accelerator` 已有抽象);`_tensor_to_bytes` 的 int16 视图假定 2 字节 dtype(fp32/int8 池会错——当前组合不可达,留注释即可)。
- **F7(nit)**:`scheduler.schedule()` 的 fcfs "empty" 返回缺 `schedule_s`(stall_free 路径有)——指标口径小不一致。

### 指标审查

- 交叉核对 README 徽章 / PLAN 冻结门 / HANDOFF / 各轮报告:**数字一致,无发现虚报**(141.5 tok/s b64、容量下界 1.070 req/s、P22 t7 默认 +0.47%、P23 0.997×、P24 1.028× 均能与证据链对上)。
- 一个**口径提示**:30B b1 decode 步时在不同文档出现 142–159 ms(PLAN §0)与 181.57 ms(P22 ⑧ 窗口)两个数——PLAN 注明"口径相关",但对外部读者不友好,建议在 PLAN §0 加一行三列小表(口径 × 数字 × 出处)。
- 方法论注记:本轮两个发现都属"**预检查与执行不同步**"类(P23 引入),恰好落在既有 SOP 第 1 条("调度时算好的形状,launch 时要按活状态复核")的延长线上——SOP 该补一句:"**预检查必须是执行的总和而非逐项**"。

---

## 4. 展示定位评估与 P25+ 探索计划(门为草案,立项时冻结)

### 4.1 展示定位评估

> 已移至本地维护文档(不入库);本文件只保留技术内容。

### 4.2 提案(P25 起,按 ROI 排序;门草案开跑前冻结进 PLAN)

| # | 项 | 内容与理由 | 门(草案) | 预估 |
|---|---|---|---|---|
| R0 | **push + 文档刷新** | **✅ 已完成(11-08)**:用户授权 push(origin = ae4e055,P19–P25 上远端);engine-vs-vllm-sglang.md 矩阵已刷新(F5) | 无(纯整理) | ✅ |
| R1 | **F1/F2 修复 + 回归** | **✅ 已完成(P25 修复轮)**:投机预检查累计化(`BlockManager.total_launch_shortfall`)+ overlap disable 排空式收尾 + NO_SPEC 门控化;`tests/p25_review_repro.py` 转为修复验证(rc=0),新增 `tests/p25_overlap_fix.py`(t1 drain bitwise/t2 翻转 ABA bitwise/t3 压力完成);台账 ISSUES I36/I37/I38 | ①repro rc=0 ✓ ②p25 三门 ✓ ③四套回归+p23 全套 ✓(见测试链日志) | ✅ |
| R2 | **overlap 服务端产品化** | **✅ CLI 已补**(`--scheduler-overlap`,api_server 透传);在线口径补测(HANDOFF Top-N #2:错峰+长队列 p95/吞吐对照)待下一轮 GPU 窗口 | 同 P23 门族;在线 A/B 补 p95 | CLI ✅ / 在线补测待排 |
| R3 | **量化路径立项(最大功能缺口)** | 平台 fp8 no-op → 做可行的那条:**int8 weight-only 加载 + 运行时反量化**(W4A16/AWQ 亦可),或 SmoothQuant 式 W8A8(先显微探针验 MUSA int8 bmm 数值);同时产出《为什么 MUSA 上 fp8/投机不可行》深挖文档(零 GPU 成本,讲解叙事) | 预注册:①加载与 bf16 参考 LCP 翻转 ≤0.5% ②吞吐损失 ≤15% ③显存 ≤60% ④四套单测 | 1–2 轮 |
| R4 | **radix-tree 前缀缓存** | 对齐 SGLang 叙事;与现有 block-hash 做同负载 A/B(块内部分命中场景) | ①bitwise 不变 ②多轮对话负载前缀命中 tokens 提升 ≥20% ③TTFT 改善可测 | 1 轮 |
| R5 | **Grafana 面板 + README 截图** | /metrics 已有 Prometheus;做一个 dashboard JSON + 截图入 README——演示成本低、观感收益高 | 面板含 SLO 直方图/队列/抢占/KV 利用率 | 半天 |
| R6 | **xgrammar 补全**(Top-N #3) | json_object(builtin)/guided_choice;bitmask 展开下沉 triton-mtgpu | 同 P24 门族 | 半轮 |
| R7 | **工程讲解文档 WALKTHROUGH.md(本地)** | 每模块 5 分钟讲解稿 + "为什么否决 X"决策清单 + 常见追问清单(现有 engine-vs/ISSUES/CLOSEOUT 是现成素材) | 无 | 1 天 |
| R8 | **PD 线收尾**(Top-N #6) | PD 流式输出/多实例;KV int8 传输(P18-D,注意 I20 终判仅限"本模型运行时",传输路径不受影响) | 同 P15/P18 门族 | 1 轮 |
| R9 | 性能对标刷新 | 用现栈重跑 vllm-musa 基线(P0 的 8.4 tok/s 已旧),产出诚实 head-to-head 图 | 交错 ≥5、中位数 | 半轮 |

**不建议做**(维持既有判决):graph replay(封存)、int8 KV 运行时(终判)、EAGLE3、多模态/LoRA 堆功能、多机调度(硬件不可验证)。

### 4.3 用户裁决项(沿用 HANDOFF §D,未变)

① push;② 厂商工单;③ P16-A 默认翻转;④ 本文档 §4 计划取舍。

---

