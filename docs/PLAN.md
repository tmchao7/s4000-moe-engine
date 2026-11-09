# PLAN — 当前迭代计划与冻结门(唯一清单)

> **⛔ 项目已冻结(P26 收官轮,见 [CLOSEOUT.md](./CLOSEOUT.md) §E 停止线)**:本计划不再排新轮次,复活条件与独立项目候选见停止线一节。下文保留为历史判决与冻结门的索引,供引用与回归。
> 历史轮次(P0–P21)报告在 `benchmarks/archive/`,当前轮(P22 起)在 `benchmarks/`,判决摘要见 [HANDOFF.md](./HANDOFF.md) §1;
> 规则:每个实验先写门再跑;一次只跑一个 GPU 任务;失败/负结果照样入报告。

## 0. 当前栈与已判决(不要重测)

**生产栈**:`attention=triton_paged`(默认)、`moe_backend=grouped_mm`(prefill)、`decode_moe_backend=bmm_dense`、`enforce_eager=True`、`scheduler=stall_free`(在线)/`fcfs`(离线)、TP2/EP2、chunk-256。30B eager b1 decode 步时 142 ms(口径相关;**P25 口径注记**:文档中另有 181.57 ms 一数,出自 P22 ⑧ 的 30B 单载窗口,含采样面、非同一负载口径——引用时注明出处,勿跨口径对比);**分段图已终局(P12 在线 FAIL + P13 修复验证:padding 税证实并消除,残余 AR 串行化 +1.0~4.7% 超 F2 门)→ 定格为单流可选开关,服务化分支关闭**;离线 b32 133 tok/s;**生产栈容量下界 1.070 req/s(1.6× 档双口径 100%,P13)**;短 prompt 四档 Poisson 全 100% SLO(TPOT p95 0.26–0.29);**1k 长上下文:冷缓存需 chunk-1024(0.5 档 90.6% SLO 过门),暖缓存任意 chunk 100%(P13)**。

**已判决(负结果也是结论)**:投机解码 NO-GO(α≈0 且 c=0.39);**I19 判决(P11):b1 固定等待主体 = 逐算子 host launch 开销,分段 replay 可回收 36.2 ms/步**(单算子融合被否定、replay 有效——两条自洽);整图 graph 否决(I29:TP2 引擎级间歇性 MCCL replay 楔死,barrier 无效),**分段图(图内零 MCCL)离线验证成立但在线否决(P12)+ 修复验证后仍转正失败(P13:padding 税已除,残余 AR 串行化 +1.0~4.7%)→ 服务化分支关闭,定格单流开关**;**I30 关闭(P13 ABA:现树快于 pre-P8 栈 23–51%,P9 回归主张被否定)**;**I8 长上下文闭合(P13:冷缓存塌=chunk 粒度,chunk-1024 解;准入=稳定性)**;**split=0 与 chunk-2048 默认化均否决(P10 实测)**;profiled placement 不采用;fused RMSNorm / bmm 容量收缩不采用;host-sync 移除与 mixed-split 已采纳默认开;30B 首 batch 为一次性 warmup 态(I31,token 门必丢首 batch)。

## 0a. P22 采样面补齐(2026-11-08 立项;对齐 vLLM/SGLang 服务层;用户裁决只做 A,B/C 入 backlog)

| # | 项 | 门/判定(预注册,不可改) | 状态 |
|---|---|---|---|
| A | **采样面补齐**:top_p/top_k/min_p + presence/frequency/repetition penalty + stop_token_ids(引擎级) + logprobs(top-k) + skip_special_tokens + `/v1/completions`(流式+非流式) | ① t1/t1b/t3 掩码与 penalty 对独立 CPU 参考逐位 ② t2 支撑集封闭(200 抽 0 泄漏) ③ **t4 全默认与逐字 legacy 体 bitwise 相等(硬门)** ④ t5 0.6B 默认 greedy 与改造前基线 bitwise ⑤ t7 默认步时 ≤±1%;top_p 激活 ≤+10% ⑥ API 冒烟 6 项 ⑦ 回归(四套+特性套) ⑧ 30B 单载窗口(一致性+步时 CV+旋钮 sanity) | **判决 PASS(11-08)**:①–⑧ 全过(t5 基线逐位全等;t7 默认 +0.47%、top_p 激活 1.091 贴门照录;API 冒烟 6/6;回归 8/9 套绿,p16-priority 引擎门=P16-C 已照录负结果;⑧ 30B 单载:greedy 自一致逐位 True + 步时 181.57ms CV=0.0004 + 旋钮/stop/logprobs sanity 全 True);**全默认走逐字 legacy 路径,per-request opt-in,生产栈零默认变更**;报告 benchmarks/P22-sampling-face.md |
| B | overlap scheduler(vLLM V1/SGLang 同款;schedule+prepare 与 GPU forward 重叠) | ① 稳态 bitwise(greedy+采样同种子) ② 错峰到达翻转 ≤0.5% ③ 抢占回退:全完成+块账一致 ④ 步时 ABA 双档 CV≤5% ⑤ 回归 ⑥ 30B 单载 | **判决 PASS(11-08)**:①②③⑤⑥ 全过(t1/t2 逐位、t3 0.347%、t4 pre=3 12/12、30B bitwise);④ b1 0.9972/b16 0.9978 小幅为正。**I36 根因修复**:投机行 launch 前被追加 1 token,`len%256==1` 块边界判断差一块 → slot 越界写+kernel 越界读 → MUSA 提交卡死(2/2);修复=按 launch 时刻长度做块覆盖检查;E1 教训=块分配时序实验必须带 bitwise 门。opt-in `scheduler_overlap=True`,默认串行逐字保留;报告 benchmarks/P23-overlap.md |
| C | 结构化输出 xgrammar | ① bitmask 语义 vs 官方 CPU apply ② schema 合法率 100% ③ regex 终止 ④ 不约束 bitwise 不变 ⑤ 开销 ≤10%(P27 起 informational;`--perf-strict` opt-in) ⑥ 校验/降级 ⑦ API 冒烟 | **判决 PASS(11-08)**:①支撑集逐位相等 ②四 schema 16/16 合法+grammar 终止 ③fullmatch ④中性路径 ⑤1.028×(历史判决;P27 g5 x5 recorded,默认不进 rc) ⑥400/降级 ⑦8/8;`guided_json`/`guided_regex`/`response_format json_schema` per-request opt-in;报告 benchmarks/P24-guided.md |

其余 backlog:n>1 并行采样、seed(MUSA RNG 语义厂商风险)、tools、swap 抢占/KV offload、PD 流式/多实例、json_object/guided_choice、bitmask triton-mtgpu 内核。

## 0b. P27 验收修复轮(2026-11-08;冻结态缺陷修复,不新增功能)

独立验收审计(P26+ 交接包)判 A 机械门 PASS、B/C/D/E FAIL,本轮逐项修复并复验:

| 项 | 修复 | 验证 |
|---|---|---|
| **I39 overlap stale prefill 行越界(high)** | `_prune_batch` 用冻结 `prefill_flags`;`num_cached >= num_prompt_tokens` 的已完成 prefill 行丢弃,不再发 1-token prefill 行 | `tests/p25_overlap_prefill_boundary.py` C1(512/768/1024+chunk 256/512);旧代码失败、新代码通过 |
| **I40 冻结记账(medium)** | `ScheduledBatch` 冻结 `prefill_flags`;`postprocess/_finish_step` 只用冻结 `query_lens/prefill_flags`(vLLM V1 同规则) | 同上 C2:200-token 中途到达后 num_cached 记 200(旧代码 1) |
| **S4 测试语义** | p24 g2/g3 改 greedy(采样可合法空白循环到 max_tokens);p24 g5/p16pri 15% 锚点/TTFT 5×/p22sam e2e 全部 informational-only;`p23/p25 --stage` 失败返回 rc=1 | p24 单跑 PASS;强门 16/16 |
| **D1 pytest 干净 checkout** | `test_benchmarking_v3.py` 指回 `scripts/archive/p13_*`;`remote_regression.sh` 镜像同步(先删目录再解包) | 干净服务器树 `pytest 28/28`(无 `scripts/p13_*.py` 残留) |
| **公开复现链(E1–E3)** | p16lru/p16pri/p24 g1 支持 `--model`;runner 传 `--model/--out`;pyproject 增 `test`/`plot` extras;SETUP 改 `pip install -e ".[test,plot]"` | 服务器 16 套全跑通;SETUP 命令链可执行 |
| **文档/脱敏(D2–D11,E4–E9)** | t5 数字、TEST-MATRIX 15%/18 行/抢占行、CLOSEOUT I28、README archive/台账指针、ISSUES E1/死路径、PLAN I30/159ms、torch 版本、AGENTS 链接、面试/简历文案、`.gitignore` 注释 | `check_docs` rc=0;逐条对照证据源 |

回归入口扩为 **16 套**(新增纯 CPU `boundary` 门);服务器强门 **16/16 rc=0**,pytest **28/28**;本地 fuzz 60 组/3048 行 0 违规。生产默认栈未变,项目仍为冻结态。详见 `docs/ACCEPTANCE-FIX-REPORT.md`(本地)。
### P29 候选(评估,未执行)

独立评估 + 外部引擎调研 + 9 天候选任务见 [ROADMAP-NEXT.md](ROADMAP-NEXT.md);Day1/2/3/4/5/6 实测见 [benchmarks/P29-day1.md](../benchmarks/P29-day1.md)、[benchmarks/P29-day2.md](../benchmarks/P29-day2.md)、[benchmarks/P29-day3.md](../benchmarks/P29-day3.md)、[benchmarks/P29-day4.md](../benchmarks/P29-day4.md)、[benchmarks/P29-day5.md](../benchmarks/P29-day5.md)、[benchmarks/P29-day6.md](../benchmarks/P29-day6.md);收口/面试口径见 [P29-FINAL.md](P29-FINAL.md);长上下文见 [benchmarks/P29-day8-longctx.md](../benchmarks/P29-day8-longctx.md);多副本运维见 [benchmarks/P29-day9-dp2-ops.md](../benchmarks/P29-day9-dp2-ops.md)。当前生产默认不变。

### P28 产品化收尾(11-09)

| 项 | 变更 | 验收 |
|---|---|---|
| 产品入口 | `s4000-server` console script;CLI/env 统一;默认生产栈;loopback 默认 + API key | `--help`/`--version`;contract tests |
| API 产品面 | 非流式去 `metrics`;可选 Bearer;`/health` version;日志去 benchmark 文案 | 10 个 API/CLI 契约测试;0.6B smoke |
| 部署 | install(venv/--target fallback)/serve/systemd/smoke/release | install 实跑;systemd-analyze verify;0.6B smoke |
| 发布包 | slim tar 排除 benchmarks/tests/assets/dev docs | 构建脚本 + 内容断言 |
| 文档 | README 部署优先;DEPLOY/OPERATIONS/DEVELOPMENT/BENCHMARKS;benchmarks 标开发证据 | check_docs rc=0;发布包阅读路径 |
| 验证 | 0.6B smoke、install fallback、API 契约、全量回归 | pytest 38/38;16/16 rc=0;服务器 0 MiB |

不改引擎性能/默认数值路径;MUSA wheel 与模型权重仍由用户准备;不做 Docker/K8s/多租户。
**P27 review 补(2026-11-09)**:独立 code review 在 `ef23c59` 上发现并修复 1 个 high 回归——初版 I39 用 prompt 长度判 prefill 完成,被抢占 seq 重 prefill 时会被误丢;改为 live `is_prefill` 完成信号 + scenario C4。同时补 p24 g5 的 PASS/DRIFT/INVALID 记录语义(`--perf-strict` opt-in)、p16pri 离散 first-token 门、p22sam `--e2e`、rsync 镜像同步、jsonschema/repo extras、ISSUES 计数;并修 review 二轮 high:overlap 兜底 sync schedule 把「在飞完成 prefill 但留在 waiting」的 seq 当 q=1 prefill 发射(补 C5)。复验:本地 C1–C5/fuzz/pytest 28/check_docs;服务器 **16/16 rc=0**(p24 g5 记录 INVALID,默认不进 rc)。

## 1. P16–P18 迭代内容(2026-11-07 立项;用户三方向合成:统一混合 batch + KV 调度做足 → KV int8 量化 → PD 小模型推进;30B 只做点睛)


### P16 引擎做足

| # | 项 | 门/判定(预注册,不可改) | 状态 |
|---|---|---|---|
| A | **统一混合 batch**(vLLM V1/SGLang MIXED 形态;现状=`_run_mixed` 双 pass,I19 launch-bound) | ① 数值:混合步 LCP 翻转率 ≤0.5% vs legacy split(0.6B paged,同引擎同历史 I34);不过则 `NANOVLLM_MIXED_MOE=split` 再测,记录哪版过门 ② 性能:混合步 wall ≤0.9× legacy(0.6B+8B;CV≤5%,交错≥5,ABA) ③ 在线:混合负载 ITL p95 ≤ legacy(照实记录),吞吐降幅 ≤2% ④ 正确性:tests/p16_mixed_unified.py(纯 decode/prefill 步不受影响;fcfs 不产 mixed;边界组)+四套单测 ⑤ 门全过→默认 `NANOVLLM_MIXED_UNIFIED=1`,否则 opt-in 照录 | **判决 opt-in(11-07)**:② **PASS 双档**(0.6B 0.749×/8B 0.850×,CV≤0.044,暖机段修复 rep1 JIT 伪象)③ **PASS**(交错×3 pooled p95 0.976,吞吐 1.007;单次 1.079 系噪声)④ 单测 PASS(结构全过;7/8 逐位)① **FAIL@0.6B**(0.5602%>0.5%,确定性;**split 回退同值**→翻转源=GEMM 批组成而非 MoE 后端,回退路径穷尽;**8B 0% PASS**;对照:legacy 自身 burst-vs-drip 历史敏感性 0.294%@0.6B/0.285%@8B=同类现象;业界同机理,vLLM batch invariance 亦 beta opt-in 默认关)→ 按⑤保持 opt-in `NANOVLLM_MIXED_UNIFIED=1`,默认 legacy split 不变;报告 benchmarks/archive/P16-mixed-unified.md |
| B | **前缀缓存 LRU 淘汰**(现状零淘汰,池耗尽=准入跳过/抢占) | ① 单测:只淘 ref_count==0 的 hash 块 ② 活动序列生成 bitwise 不变(0.6B paged 同引擎同历史) ③ 压力负载准入硬悬崖消除(开关 A/B 对比 429/skip 曲线,seed-per-repeat) ④ 主动复用模式前缀命中率不降 ⑤ 四套单测;默认开,`NANOVLLM_LRU_EVICT=0` 关 | **判决 PASS(11-08)**:两级 victim(fresh 优先+LRU 冷块,touch-on-hit);①⑤ ✓(双策略模糊测试)②bitwise ✓ ④✓;**③前提修正**:cached 块本就住 free 列表(revive 出列),准入悬崖=纯容量,缓存从不扣留块;顺序流量 FIFO≡LRU(结构使然),分歧窗口=排队命中者 touch(t2 确定性证明);报告 benchmarks/archive/P16-lru-priority.md |
| C | **优先级/SLO 分级调度**(SamplingParams.priority;插件点=waiting 候选/preempt victim/chunk 顺序) | ① 0.6B 混合负载 高:低=1:3,高优先 ITL p95 ≤0.8× 无优先级基线(同引擎 A/B) ② 低优先全部完成(饥饿守卫 32 步保留) ③ 四套单测 | **判决:特性采用,门①FAIL 照录(11-08)**:priority 默认 0=逐字节 legacy(零回归面,per-request 无全局默认可翻);**高优 TTFT p50 21.8s→0.511s(42.7×)**(真实抢占压力 3 次/臂);ITL p95 比值 0.996(收益机制=准入延迟,非 decode 干扰);低优不饿死 ✓;顺带修复 I35(stall_free for-range 抢占越界,上游潜伏);api_server 透传 priority;报告 benchmarks/archive/P16-lru-priority.md |
| D | **30B E1 单载 e2e**(tuned 表 `NANOVLLM_PAGED_CONFIG=auto` vs 默认;一次加载窗口) | e2e/decode 时延 ≥5% 且 CV≤5%(ABA 交错);负结果照录 | **判决:负结果(11-08)**:e2e ratio 1.010~1.020(3 rep,CV<2%,极稳)——auto 一致性略慢;内核级 2× 被 MoE 主导步时稀释;tuned 表维持 opt-in,生产默认 (32,4,1) 获 e2e 确认;**E1 项闭合**;报告 benchmarks/archive/P17-kv-int8.md §E |

### P17 KV int8 量化(I20 立项;探针:独立 dequant pass 慢 5.7×,必须融 kernel)

| # | 项 | 门/判定(预注册,不可改) | 状态 |
|---|---|---|---|
| A | **int8 KV 池+量化写 kernel**(opt-in `kv_cache_dtype="int8"`;scale `[L,blocks+1,256,kvh]`;block_bytes 减半) | ① 量化 kernel 与 torch 参考逐位一致 ② 池块数 ≥1.8×(同 gpu-util) ③ 四套单测 | **判决 PASS(11-08)**:① BITWISE(正确池形状复核)② **2.0× 精确**(730→1460 块,block_bytes 减半)③ 全绿 |
| B | **读侧融合 dequant**(paged decode kernel 两处 load 后乘 scale;`_gather_kv` gather 后 dequant) | ① LCP 翻转率 ≤2%(int8 有损,per-token-head absmax 探针往返误差 0.67%)+32-prompt 生成 sanity(完成率 100%,无乱码,平均长度偏差 ≤10%) ② decode 步时 ≤+5% vs bf16(CV≤5% 交错≥5) ③ 容量曲线:1.6× 档并发吞吐不塌 ④ 四套单测;门全过仍默认 bf16(opt-in) | **判决:机制 PASS,数值门 FAIL(11-08)**:② decode 步时 1.013× ✓ sanity ✓ ① **FAIL — 0.5(32/32 首 token 分叉,生成退化)**;**根因实测**:Qwen3 K per-head amax 196~492 vs 典型元素 0.5 → per-token-head absmax 步长淹没信号(逐件排除 plumbing:store/gather 逐位、paged+scale 4.8e-4、引擎内重算 0.0019);**P19 修复尝试 per-channel 校准:机制逐位正确但质量门仍 FAIL(通道内动态范围 500× 超 int8 承载,fp8=MUSA 静默 no-op)→ 混合精度 outlier 通道下轮立项;默认 bf16 不变;ISSUES I20;报告 benchmarks/archive/P17-kv-int8.md 与 P19-i20-review.md |

### P18 PD 小模型推进(吃 P16/P17 产出;M1/M2/M3 已 POSITIVE)

| # | 项 | 门/判定(预注册,不可改) | 状态 |
|---|---|---|---|
| A | Poisson 持续负载 collocated vs PD A/B | 同款判决:PD ITL p95 < collocated;补 TTFT 对比 | **POSITIVE(11-08,双档)**:0.6B qps0.5×3rep **ratio≈0.79**(0.1023~0.1048 vs 0.1213~0.1311)、8B qps0.3×2rep **ratio≈0.82**(0.121 vs 0.148);PD TTFT +1.2~1.9s(交接传输已知代价);**PD 赢 SLO 结论在闭式突发/0.6B Poisson/8B Poisson 三种负载形态下全部成立**;脚本 scripts/archive/p18_pd_poisson.py |
| B | PD OpenAI 兼容端点(`/v1/chat/completions`) | 与 collocated api_server 同负载 A/B;冒烟客户端过 | 未跑 |
| C | P 侧前缀复用 + D 侧优先级(吃 P16-B/C) | 重复前缀请求 P 侧 TTFT 下降可测;D 优先级门同 P16-C 口径 | 未跑 |
| D | KV int8 传输(吃 P17;kv_channel index_select/index_copy_ 路径) | 传输字节 ≈½ 且传输时间下降;D 终态数值门同 P17-B;汇总组合收益 | 未跑 |

### P19(2026-11-08 迭代轮:I20 修复尝试 + 全面 code review + 指标核查)

| # | 项 | 门/判定 | 状态 |
|---|---|---|---|
| A | I20 修复:per-channel 校准静态 scale | 逐位 unit + 引擎连贯性/翻转≤2%/decode≤+5%/容量 2× | **机制逐位正确(store BITWISE/引擎内全行恒等/bulk 60 级);质量门 FAIL — 通道内动态范围 500× 超 int8 承载;fp8=MUSA 静默 no-op;混合精度 outlier 通道下轮立项**;报告 benchmarks/archive/P19-i20-review.md |
| B | 全面 code review(独立 agent 审 8 文件) | HIGH bug 修复+回归测试;风险/nit 全处理 | **修复:分块重排队×优先级=同 seq 单步重复调度(HIGH,t6 回归)+wait_steps 语义+校准异常安全+计数器重置+api 400+touch 泄漏+scale hoist**;7/7 回归全绿 |
| C | 指标核查 | /metrics 新计数器 + priority 400 | **✓ 活体冒烟**(kv_lru_size/kv_prefix_hit_tokens_total 暴露;priority 非整数→400) |

### P20(2026-11-08 迭代轮 2:I20 终局 — 混合精度 outlier 通道)

| # | 项 | 门/判定 | 状态 |
|---|---|---|---|
| A | 混合精度 int8 KV(per-(layer,kv_head) bf16 outlier 维 ≤4 + 侧池 + split-sum kernel) | 同 P17-B 五门 | **机制逐位正确(unit 四张量 BITWISE; 引擎重算 0.0033; 修复 P19 遗留: paged 读误用 store scales)**;**质量门 FAIL(flip 0.5/sim 0.035/degenerate 2) — 三方案连续 FAIL → I20 终判: 本模型 K/V 动态范围对 int8 不可用**; 容量 2× 奖池放弃(除非厂商 fp8); 默认 bf16 不变; int8 代码保留 opt-in-experimental |
| B | P16-A 默认翻转评估: 30B 单载证据(统一 vs legacy, 信息性门) | ratio ≤1.0 + CV≤5% + flips=0 | **EVIDENCE PASS(11-08)**: ratio **0.977**(−2.3%, 3 rep CV<0.1%) + **flips=0 逐位全等**;P16-A 证据链完整(0.6B 性能 0.749×/8B 0.850× 逐位 0 翻转/30B 0.977× 逐位 0 翻转;唯一瑕疵=0.6B 翻转 0.56%);冻结门①钉在 0.6B 不可翻,默认维持 opt-in,**门修订提案留给用户裁决**;脚本 scripts/archive/p20_30b_flip.py |
| C | P18-C: 优先级×PD(ctrl 透传 + D 侧准入背压 + P 侧前缀复用) | 复用门:重复 prompt prefill < 0.6×;压力门:全员完成+anchor | **PASS(11-08)**:**前缀复用 prefill 0.835→0.146s(5.7×)**;priority 全链路到 D 侧 Sequence(admission 日志实证);修复 PD 控制面两处 bug(bulk 帧先于准入的 popleft 崩溃 + 单 chunk 多帧失衡 → 双 FIFO 平衡队列);D 侧优先级差异化中性(如实记录;强证据在 P16-C);anchor 改统计口径(PD 网络抖动,I34 bitwise 锚不适用) |
| D | P18-B: PD OpenAI 兼容端点 | 冒烟:health + chat completion 连贯 | **PASS(11-08)**:scripts/archive/p21_pd_openai.py(FastAPI 非流式 + priority 透传 + pd_metrics prefill_s/ttft 暴露);活体冒烟 ✓ |

### 范围注记

- P16-A 本阶段**不做**单 varlen paged 内核(q_len≥1 统一 attention,SGLang 式)——列为 A2 可选后续,需过 E1 同款翻转门。
- P15 row E 的 backlog(KV LRU/优先级/I20)由本表 P16-B/C + P17 承接。
- PD A/B 的 collocated 基线在 P16-A 落地后用同版本引擎重新对臂。

## 1a. P15 收官存档(2026-11-07 立项,合成一条线:快赢 → I29 复测 → PD 1P1D;新实例 s4000,栈=torch 2.11.0.post1+triton 3.2.0+transformers 5.13,四套单测+30B 冒烟已全绿)

| # | 项 | 门/判定(预注册,不可改) | 状态 |
|---|---|---|---|
| A | **E1 paged kernel autotune** | ① 数值:非 bitwise → L2 翻转门(<0.5%)+首 token 100% ② 性能 ≥5% CV≤5% ③ 四套单测 | **判决 FAIL(11-07)**:内核 1.59–2.90×(b1→b32 geomean ~2×)但无全局 bitwise 配置,0.6B LCP 口径翻转率 0.81%>0.5% 门 → **默认保持 (32,4,1),tuned 表降级 `NANOVLLM_PAGED_CONFIG=auto` opt-in**;报告 benchmarks/archive/P15-pd-autotune.md §A |
| B | **I29 新栈复测** | 引擎级 0.6B TP2 paged replay ×3 判定 | **判决:维持(11-07)**——纯张量梯子 7/8 绿;引擎级 2 clean + **1 WEDGE(init/capture 段,驱动级无主残显存 22GB,需重启实例)**。形态从"确定性首 replay 楔"变为"间歇性 init 楔"→ 30B graph 维持禁止;素材已入厂商工单 |
| C | **PD 1P1D 主线** | Step 0 ITL 2× 门 / M1 bitwise / M2 bitwise vs 同引擎对照 | **Step 0 GO**:**M1 PASS**(三臂 bitwise;I28 变体修入通道:index_select/index_copy_ 连续化);**M2 PASS @ 0.6B+8B 双档**(0.6B: 57ms/14-21ms/1.27s;8B: 74ms/18-27ms/1.4s,块=36MiB;两档 PD==同引擎 collocated 6/6 bitwise;8B 处女 baseline 也全等);**新发现 I34**:collocated token 依赖引擎状态历史(泛化 I31)→ bitwise 门必须同引擎同历史对照;下一步=8B 复验+在线 A/B(待实例重启清残显存) |
| D | (并行,零 GPU)用户亲自发送 `docs/E-vendor-ticket.md`;E2(vllm_fused 调优配置,非生产路径)降级 backlog | — | 待用户 |
| C-M3 | **PD 并发化 + collocated vs PD 在线 A/B** | 门:PD ITL p95 < collocated | **POSITIVE(11-07)**:0.6B p95 比值 **0.753×**、8B **0.839×**(全分位改善,e2e −41~48%,吞吐 −15~27% 符合"PD 赢 SLO 不赢吞吐");TP1 去 AR 混淆已注记;报告 benchmarks/archive/P15-pd-autotune.md §H;PD 主线 M1/M2/M3 全闭环 |
| E | backlog(过 A–C 后再排):KV cache 淘汰(LRU)、优先级/SLO 分级调度、I20 KV int8(须融 kernel) | — | 未排 |

## 1b. P13 收官存档(2026-10-30 五项全判决,细节见 git 历史与本节报告)

I30 关闭 / I33 修复验证后段图定格单流开关 / I8+I26 闭合(chunk 粒度) / 容量下界 1.070 req/s / 厂商工单成文。报告:`benchmarks/archive/P13-*.md`。


## 2. 冻结门汇总(集中一处,**均不可改**)

| 门 | 位置 | 阈值 | 状态 |
|---|---|---|---|
| attention 等价 | `docs/numerics-policy.md` 五层门 L1–L5(实现 `scripts/p6_decide_attention_equivalence.py`) | L1 max_abs<2e-3;L2 翻转率 <0.5% 且翻转处 margin <0.5;L3 首 token 100%+退化 0+人工判读;L4 双端自确定;L5 在线 SLO | ✅ paged 已转正;`scripts/p5_decide.py::ATTENTION_GATES` 为历史保留,阈值不改 |
| graph 转正(双轨) | `scripts/p5_decide.py::GRAPH_GATES` + P9-graph.md §4(P11 修正:30B token 门丢首 batch,I31) | token 全等(b1/b8/b32,稳态)+ CV ≤5% + 显存增量 ≤2 GiB;b1 ≥+15% 转正 | **离线五项全过(b1 −23.7%),但 P12 在线 A/B FAIL → 不转正**;整图维持否决(I29);门本身需补非桶边界 batch(P12 教训) |
| MoE 后端采用 | 已过期(历史门) | b16/b32 吞吐比 ≥1.08、CV ≤5% | ✅ bmm_dense 已过 |
| 在线 SLO(双口径) | v3 口径,逐请求从 records 复算 | standard: TTFT ≤2.0 s 且请求级 TPOT ≤0.5 s;strict: + ITL ≤1 s | 生产门;v4 回归门 TPOT p95 ≤0.343 现树 0.26–0.29 达标(I30 已关闭) |
| 回归完整性 | `tests/p2_grouped_mm.py`、`tests/p6_bmm_dense_tiny.py`、`tests/p8_scheduler_gaps.py`、`tests/test_benchmarking_v3.py` | 六配置逐 token 全等 rc=0;调度单测 6/6;v3 单测 28/28 | 每轮开工先跑(服务器) |
| 长上下文改进(历史) | P9-long-context.md | 1k 50% 档 TTFT p50 ≤2 s | ✅ chunk-2048 达成(1.036 s);后续回归基线 |

## 3. 红线与 SOP

全部集中在仓库根 `AGENTS.md`(本地 agent 入口,公开仓库不含;公开读者按 [SETUP.md](./SETUP.md) 与 [HANDOFF.md](./HANDOFF.md) 操作),本文不重复。

## 4. 文档地图

想知道什么 → 读什么:`AGENTS.md`(入口/命令/红线)→ `docs/HANDOFF.md`(状态与判决)→ `docs/PLAN.md`(本文,迭代与门)→ `docs/ISSUES.md`(问题台账)→ `docs/architecture.md` / `numerics-policy.md` / `musa-pitfalls.md`(专题)→ `benchmarks/`(当前轮报告,历史在 `archive/`)。
