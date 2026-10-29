# P13 任务书 — 硬骨头攻坚(交接给下一个执行 Agent)

> 这份文档是**下一轮的唯一任务书**。它自包含:环境、命令、已判决清单、五个攻坚项的假设与冻结门、红线、验收。
> 读完本文再读 [HANDOFF.md](./HANDOFF.md)(状态)、[PLAN.md](./PLAN.md)(门)、[ISSUES.md](./ISSUES.md)(台账);本文与三者冲突时,**以本文的实验设计为准,以 ISSUES 的事实为准**。
> 用户诉求(2026-10-29):这个项目已经"做完"了(P0–P12 全判决),但还剩几个**硬骨头**;不要为了翻默认再烧时间,目标是**把问题查清、把便宜的数字拿到**。

---

## 0. 三十秒状态

- **分支** `feat/musa-port`,HEAD 见 `git log --oneline -3`(本地,未 push;push 需用户明确授权)。
- **服务器**:idle,双卡 0 MiB,无进程,shm 干净。代码 `/root/moe-server-s4000`(伪 `.git`,**绝不在远端做 git 操作**)。
- **生产栈**:`triton_paged` + `grouped_mm`(prefill)/`bmm_dense`(decode) + `enforce_eager=True` + `stall_free` + TP2/EP2 + chunk-256。**P12 判定后默认值一个没动。**
- **最近一条判决**:分段图(segment graph)在线 A/B **FAIL**(TPOT p95 比 eager 慢 5.9–14.7%),默认不启用;离线 b1 −23.7% 仍有效但**只对单流**(见 §2 陷阱 3)。

## 1. 环境与命令(照抄,别改名)

```bash
# 连接(key 认证可用;密集重试会撞 sshd MaxStartups,冷却 60–90 s)
ssh -p 28036 root@connect.sha1.seetacloud.com

# 本地→远端同步(不要用 scp)
tar czf - nanovllm tests scripts | ssh -p 28036 root@connect.sha1.seetacloud.com \
  'tar xzf - -C /root/moe-server-s4000'

# 开工核验(必须;发现陌生进程只记录不杀)
ssh -p 28036 root@connect.sha1.seetacloud.com \
  'pgrep -fa "^/usr/bin/python3|^python3|^[s]pawn_main" || echo idle; mthreads-gmi | grep MiB'

# 回归四套(全 rc=0 才允许加载 30B)
cd /root/moe-server-s4000
PYTHONPATH=. python3 tests/p2_grouped_mm.py            # 六配置全等
PYTHONPATH=. python3 tests/p6_bmm_dense_tiny.py        # 六配置全等
python3 -m pytest tests/test_benchmarking_v3.py -q     # 15 passed
PYTHONPATH=. python3 tests/p8_scheduler_gaps.py        # 6/6
```

模型与旧树:

| 路径 | 用途 |
|---|---|
| `/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8` | 30B 生产模型 |
| `/root/autodl-tmp/models/Qwen3-0.6B`、`Qwen3-1.7B` | 廉价探针(0.6B 加载 ~1–2 min) |
| `/root/models/Qwen3-8B` | 中档 dense |
| `/tmp/tiny_moe` | 单测用 tiny MoE |
| **`/root/moe-server-p6d`** | **pre-P8 旧树(`d5779b6`,P11 已验证可起服务)**——P0 用,用完删 |

生产栈启动(30B):

```bash
cd /root/moe-server-s4000
NANOVLLM_TRACE_INIT=1 PYTHONPATH=. python3 nanovllm/entrypoints/openai/api_server.py \
  --model /root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --decode-moe-backend bmm_dense --max-num-seqs 64 > /root/logs/api.log 2>&1 &
```

- **`--decode-moe-backend bmm_dense` 不可省**:默认 None 跟随 `--moe-backend`(grouped_mm),那样跑的不是生产栈(I32-①)。
- 段图另加 `NANOVLLM_SEGMENT_GRAPH=1 --no-enforce-eager`,且 `--max-num-seqs` 必须 ≤64(否则 3360 张图捕获期 OOM,I32-②)。
- 判活 = `/health` 200 **且** 日志出现 `[init] handshake_done`。热缓存 ~7 min,冷缓存 11–12 min(FP8 逐 expert 反量化 5.6–6.7 min)。

在线矩阵(冻结口径,勿改):

```bash
PYTHONPATH=. python3 scripts/benchmark_online.py \
  --model Qwen3-30B-A3B-Instruct-2507-FP8 \
  --mode poisson --max-qps 0.87 --fractions 0.5,0.75,0.9,1.0 \
  --requests 128 --prompt-words 128 --repeats 3 --request-timeout-s 120 \
  --output /root/p12_data/<arm>.json
```

A/B 判定器(纯离线比较两份 JSON,三门判据见 `benchmarks/P12-serving.md`):

```bash
python3 scripts/p12_serving_ab.py armA.json armB.json --verdict-out verdict.json
```

---

## 2. 已判决:不要重做(重做=浪费一轮)

| 方案 | 判决 | 出处 |
|---|---|---|
| 整图 graph(TP2) | 否决:图内 MCCL 间歇性楔死(0.6B 最小复现 3/3) | I29 |
| 投机解码朴素 draft | NO-GO:α≈0.006、c≈0.385,最优 0.73× | S1 |
| **分段图在线转正** | **FAIL:TPOT p95 慢 5.9–14.7%,默认不启用** | P12 / I33 |
| split=0(关混步拆分) | 否决:低载无收益,高载 SLO 崩到 22.7% | P10 |
| chunk-2048 转默认 | 否决:TPOT p95 +5.3~9.5% 超门 | P10 |
| fused RMSNorm | 否决:−0.76%;且 device −10.7 ms 步时不降 → "launch-bound→融合换步时"因果链已否 | P8 / I24 |
| bmm_dense 容量收缩 | 否决:−1.5% | P8 |
| profiled expert placement | 不采用:imbalance −99.5% 但吞吐 +0.01% | P9 |
| host-sync 移除 / mixed-split | 已采纳(默认开) | I24 / I23 |
| paged attention | 已转正(五层门 L1–L5 全过) | I4 |
| vllm_fused / mp22 | 禁用(mp22 数值错、vllm_fused 慢) | E3/E5 |

**六个陷阱**(每一个都真实坑过一轮):

1. **跨轮对比不作数**。P9 测现行栈 TPOT p95 0.387,今天同栈测得 0.2691 —— 不是代码变了,是机器状态(AutoDL 共享物理卡)。任何 A/B 必须**同 session 相邻**测量;跨时段的必须补一个复现臂(P12 靠 B2 复跑才立住判定)。
2. **桶边界是陷阱**。图按 `[1,2,4,8,16,32,48,64]` 整桶 replay;P11 的离线门只测 b1/b8/b32(恰好全在桶边界),于是 b9–b31 的 padding 税被系统性漏掉。**任何按桶捕获的路径,扫描必须含非边界点。**
3. **段图的 −23.7% 只属于 b1**。b8 +0.5%(持平)、b32 +3.8%(回退)。不要用它预测并发收益。
4. **首 batch 必须丢弃**(I31):进程内第一个 batch 与稳态逐位不同,是一次性 warmup 态跃迁。
5. **`EADDRINUSE` 是已知 flake**:`llm_engine._get_distributed_init_method` 先 bind 临时端口再关 socket,torch 随后 rebind。`p2_grouped_mm` 撞到时单独重跑即可,不要当回归。
6. **`pgrep -f` 会匹配到执行命令的 shell 自身**。用 `^/usr/bin/python3|^python3|^[s]pawn_main`;kill 前先列 PID。

---

## 3. 攻坚项(按性价比排序;每项的门先写再跑)

### P0 — I30 定案:现行栈到底有没有回归?(~1.2 h,2 次 30B 加载)

**问题**:I30 开着两轮了——"现行栈 TPOT p95 相对 v4 回归 13~16%(0.387 vs 门 0.343)"。但今天 P12 的臂 A' 用**同矩阵、同 seed、同 137-token prompt、同生产栈**测出 **0.2691**,比门低 21%。P10–P12 从未改过 eager 路径 → **I30 的绝对数字不可复现**。

**关键设计**:不要再去追"0.343 这个绝对门"——那个门绑着当年的机器状态。改成**同 session 的相对比较**,era-free:

```
旧树(/root/moe-server-p6d, d5779b6) → 现树 → 旧树(ABA 三臂)
每臂只跑 --fractions 0.9,1.0 --repeats 3(6 点,约 20 min)+ 加载
```

**判据(先冻结)**:
- 两次旧树跑互相吻合(CV ≤5%)才算有效;否则 INVALID,重跑。
- **现树 ≤ 旧树**(各档 TPOT p95 中位)→ I30 **关闭**:现行栈不比 v4 栈慢;P9 的 0.387 记入"机时噪声"。
- **现树 > 旧树 ≥5%** 且稳定 → 回归真实,进入 P8 期改动二分(那时才值得开新轮)。
- 无论结论,**把两臂绝对值一起记下**,用于解释 P9 的 0.387 从哪来。

**成本**:旧树 1 次加载(~7 min)+ 20 min;现树同;共 ~1.2 h。旧树起服务用**它自己的默认参数**(`d5779b6` 没有 `--max-num-seqs`/`--enforce-eager` flag,不要硬传),但要 `--decode-moe-backend bmm_dense`(P9-serving 的 v4 启动行也是这么带的)。记录两臂 KV blocks(预计 730 vs 723,差 1%,报告注明)。

**交付**:`benchmarks/P13-i30.md`,五字段报告;ISSUES I30 改状态。

---

### P1 — I33 padding 税实测:用 b16 vs b17 的一行差把机制钉死(~1 h,**零 30B 加载**)

**问题**:P12 把在线回退归因于"桶 padding"(并发 14–30 落在桶间),但这是**推断**,没有直接测量。这是整个图路线去留的关键证据。

**关键设计**(0.6B TP2 paged,`bmm_dense`,segment on/off):

```
batch 点:6, 8, 9, 12, 15, 16, 17, 20, 24, 32
```

最锋利的一行是 **b16 → b17**:16 是桶边界(cap=16,无 pad),17 落进 32 桶(cap=32,pad 1.88×)。如果 padding 是主因,这里应当出现**阶跃**;否则整条归因链要重写。

**判据(先冻结)**:
- `segment@17 / segment@16` ≥ **1.15**(步时)→ padding 税成立(计算量比 32/17=1.88,考虑 AR 与固定开销,1.15 是保守下界)。
- 同时 `eager@17 / eager@16` ≤ **1.10**(eager 只有 +6% 的行数),用来证明阶跃来自桶而不是别的。
- 桶边界上 segment vs eager:预期 b8 ≈ +0%~5%、b32 ≈ +3%~5%(复现 P11)。
- **若阶跃成立** → 结论:密桶修好后最好也只是"回到桶边界的水平",即 **−0.5%~−3.8%,永远不可能为正** → 图路径正式定性为"单流专用",不再投入。写进 I33 与 PLAN,关闭该分支。
- **若阶跃不成立** → padding 不是主因,回退另有来源(eager AR 与 replay 的串行化)。这时才值得单开一轮做 profile(那是新课题,不要在本轮硬啃)。

**实现提示**:现有 `tests/p11_step_ab.py` 只测 b1,需要加 batch 维度(它用 `p8_moe_30b.drive_batch` 的 b1 版本或 `tests/p11_30b_ab.py` 的 `drive_batch` 接口)。**代码先行 + 0.6B 验证,不要为此加载 30B。**

**如果哪天要复活图并翻默认(复活门,先写死)**:必须在**同 session 的 30B 在线 A/B** 里三门全过,且 **1.0 档 TPOT p95 优于 eager ≥3%**(不是持平)。理由:P12 的离线红利只在 b1,而在线并发 14–30 天生在窗口外;让"勉强持平 + 多一套捕获/显存/诊断成本"进默认,不值得。

**交付**:`benchmarks/P13-padding.md`(含 batch×step-time 表),I33 更新。

---

### P2 — 长上下文端到端:把 0% SLO 的负结果变成"修完之后怎样"(~1.5 h,1 次加载)

**问题**:I8 结论是"1k 输入在 0.435 qps 就崩,三档 SLO 全 0%,崩在 TTFT"。之后做了两件事但**从未端到端复测**:I26(KV 准入/429/HOL 跳过)、chunk-2048(只在 50% 档验证过 TTFT 1.036 s)。这是全项目最大的一块"已知负结果 + 已备好药方但没吃药"。

**口径**:
```
--prompt-tokens 1024(不要用 --prompt-words,会被分词器放大)
--prompt-mode unique --max-model-len 4096 --max-tokens 64
--fractions 0.5,0.75,1.0 --max-qps 0.87 --requests 128 --repeats 3
两臂:chunk 256(生产默认)/ chunk 512 或 1024(对照,不翻默认)
```

**判据(先冻结)**:
- 门 = P9 的改进门:**50% 档 TTFT p50 ≤2 s**;同时报 `slo_standard_rate`。
- **测量完整性(必须)**:I26 会把超 KV 的请求 **429 拒掉**。因此必须同时报
  - `slo_standard_rate`(**分母 = 被准入的请求**,不是全部),
  - `reject_429` 计数与拒绝率。
  只报前者、不报后者 = 用拒绝刷 SLO,报告会被打回。
- 若 50% 档 ≤2 s 且拒绝率 <2% → **I8 从"崩在 0.435 qps"升级为"1k 场景可服务"**,这是本轮最有价值的故事升级。
- 若不达标 → 记录真实容量(TTFT 分解:排队 vs prefill),不要再调 chunk 硬凑。

**交付**:`benchmarks/P13-longctx.md`,ISSUES I8/I26 更新。

---

### P3 — eager 生产栈的容量上界(便宜且简历直接可用,~45 min,可与 P2 同一次加载)

**问题**:现有饱和数据**只覆盖段图臂**(1.2/1.4 档 → 实测 0.924/1.036 req/s,双口径 100%)。生产栈(eager)的容量上界**从没测过**;而它在同到达率下 TPOT 更低,容量很可能更高。这是个免费数字。

**口径**:`--fractions 1.0,1.2,1.4,1.6 --repeats 3`,其余同冻结口径(max-qps 0.87)。

**判据**:报 `slo_standard ≥99%` 的**最大达成功率**(三 repeat 中位)+ 实测 req/s + max_concurrency。不设通过/否决,只报数。

**产出**:一个可以直接写进简历/文档的句子:"生产栈在 137-token / 128-output 负载下可持续 **X req/s** 且 100% 双口径 SLO"。

---

### P4 —(可选大课题,1–2 天,不建议在本轮做)KV 量化 I20

**现状**:KV 全 bf16,12 MiB/block,~8.5 GiB/rank(723–730 blocks ≈ 186k token/rank)。目标:容量 ×2、decode KV 带宽减半。

**为什么是"新课题"而不是"补完项目"**:它要新写 kernel(store 量化 + 读时反量化或 int8 dot)+ 过 L1–L4 数值门 + 在线 SLO,任何一环不过就没有收益。而且 S4000 的 int8 支持需要先确认(`torch_musa` 上有没有可用的 int8 GEMM / 转换路径),**先花 30 min 做可行性探针再决定要不要立项**。

**不要做**:norm/rope 融合(P8 已实测 −0.76%,且因果链被否定)、投机解码(S1 NO-GO)、整图(I29)、S6 EAGLE3(前置是 I19,而 I19 只在 b1 被段图部分解决)。

---

## 4. 红线(违反=毁掉一轮)

1. **不改默认值**:`enforce_eager=True`、`NANOVLLM_SEGMENT_GRAPH=0` 保持。要翻默认必须先过 §3 里的门,且 P12 的门是**否决**状态。
2. **不改冻结门阈值**:`scripts/p5_decide.py`、`benchmarks/P12-serving.md` 的三门、attention 五层门。门只能补(如加非边界 batch),不能放宽。
3. **一次一个 GPU 任务**;长任务 `setsid nohup` + 全量日志落盘(观测管道会吞崩溃现场)。
4. **代码先行**:改引擎代码 → 本地 + 单测 + tiny/0.6B 验证 → 才允许加载 30B。加载窗口只做测量。
5. **本地 commit,不 push**(需用户明确授权);远端 `.git` 是伪仓库,绝不在远端做 git 操作。
6. **不改 `site-packages`**;不用 `vllm_fused`/`mp22` 默认;不全局 `moe_backend=bmm_dense`(prefill 会 OOM)。
7. **失败/负结果照录**,不伪造成功;单问题 45 min 时间盒,写"已排除/未排除/下一步"再切下一条。
8. **收尾必做**:按 PID kill + 双卡 0 MiB + `/dev/shm` 干净 + 删 lock;证据从服务器拉回本地 `benchmarks/data/`(该目录 gitignored),sha256 写进报告。

---

## 5. 验收(A–E)

- **A(P0)**:I30 有明确结论(关闭 或 确定回归+开二分);ABA 三臂的有效性与 CV 都写在报告里。
- **B(P1)**:b16/b17 的阶跃数据在报告里;I33 从"推断"变成"实测",并据此给出图路线的最终定性。
- **C(P2)**:1k 三档 TTFT/SLO **以及拒绝率**都报了;50% 档 TTFT p50 与 2 s 门逐项对照。
- **D(P3)**:生产栈容量数字(≥99% SLO 的最大 req/s + max_concurrency)。
- **E(收尾)**:`benchmarks/P13-*.md` 五字段报告 ×3–4、ISSUES/PLAN/HANDOFF 更新、本地 commit、服务器清零、证据 sha256 入档。

**如果只有 2 小时**:做 P0 + P3(便宜且都产出可引用数字),P1/P2 留给下一轮。

## 6. 证据索引(本轮起点的关键事实)

| 事实 | 数值 | 出处 |
|---|---|---|
| 臂 A'(eager 生产栈)TPOT p95 中位 0.5→1.0 | 0.1968 / 0.2519 / 0.2372 / 0.2691 | `benchmarks/data/p12/p12_online_armA2.json`(sha256 `a25f9b52…`) |
| 臂 B'(段图)同口径 | 0.2108 / 0.2743 / 0.2707 / 0.2848(B2 复跑) | `p12_online_armB2.json`(sha256 `f966d103…`) |
| 复现性 | B' vs B2 逐档 ≤1.5%(1.0 档 −0.0%) | `p12_ab_verdict_samesession.json` |
| 离线 30B b1 | eager 152.85 ms → segment 116.63 ms(**−23.7%**);b8 +0.5%、b32 +3.8% | `benchmarks/P11-graph.md` |
| 段图臂饱和 | 1.2/1.4 档 → 0.924 / 1.036 req/s,双口径 100% | `p12_sat_armB.json` |
| 1k 长上下文(旧栈) | 三档 SLO 0%,TTFT p50 3.84/8.28/10.22 s;空闲单请求 TTFT 0.368 s、TPOT 0.144 s | `benchmarks/P6-i8-long-context.md` |
| 每步固定开销 | 0.6B b1:2,401 launches/step、13.5 µs/launch、device_busy 0.478–0.603;replay 回收 36.2 ms/步 | ISSUES I19 |

**接手第一件事**:`git log --oneline -3`、读 `docs/HANDOFF.md`,然后按 §1 跑开工核验与回归四套。**不要先加载 30B。**
