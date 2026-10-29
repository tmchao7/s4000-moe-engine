# HANDOFF — 当前交接(滚动更新;本版 = P12 判定收官,2026-10-29 晚)

> 本文是**唯一交接文档**(历史版本在 git 历史)。接手先读本文,再按 [AGENTS.md](../AGENTS.md) 核验环境与命令。规则不变:先写门再跑;一次一个 GPU 任务;负结果照录;本地 commit(不 push 除非用户授权);到点无论进度必须停下写交接、清服务器。

## 0. 30 秒状态

- **仓库**:`feat/musa-port`;P12 判定轮已本地 commit(未 push)。
- **服务器**:已清零核验(双卡 0 MiB、无进程、shm 干净)。`/root/moe-server-p6d`(P2 预部署旧树)仍保留,用完删。**ssh key 认证已自愈**(10-29 19:34 实测 `s/^BatchMode` 直连成功),无需密码;若再遇 `Exceeded MaxStartups`,等 60–90 s 再连并改用 ControlMaster 复用,不要密集重试。
- **生产栈(未变)**:`triton_paged` + `grouped_mm`(prefill)/`bmm_dense`(decode) + **`enforce_eager=True`** + `stall_free` + TP2/EP2 + chunk-256。
- **P12 结论一句话**:分段图离线 b1 **−23.7%**,但**在线 A/B 判定 FAIL**(四档 TPOT p95 比 eager 慢 5.9–14.7%),**默认不翻转**,P1 转正取消。原因=桶集 padding 税(详见 §2、`benchmarks/P12-serving.md`)。

## 1. 本轮(P12 判定轮)进展(证据 `benchmarks/P12-serving.md` §P0 判定)

| 项 | 状态 | 关键事实 |
|---|---|---|
| 开工回归四套 | **全绿** | `p2_grouped_mm` rc=0(六配置全等+round_robin+确定性)、`p6_bmm_dense_tiny` rc=0、`test_benchmarking_v3` 15 passed、`p8_scheduler_gaps` 6/6 |
| **P0 在线 A/B 判定** | **已产出 = FAIL** | 门① PASS(SLO 四档 100%,0.9 档 strict 差 1/128 内)、门② FAIL(1.0 档 B 0.2848 > A 0.2691)、门③ FAIL(0.5 档 B 慢 7.3%) |
| 臂 A'(eager 生产栈) | 完成 | 12/12 点 128/128、双口径 100%;init 与 B' 对称(723 blocks/rank、maxseq64);JSON sha256 `a25f9b52…48d604e8` |
| 臂 B2(段图复跑) | 完成 | 与 19 h 前的 B' **逐档重合**(1.0 档 0.2848 vs 0.28481)→ 排除机时污染;JSON sha256 `f966d103…178d683` |
| 归因 | 机制已定位 | `graph_bs=[1,2,4,8,16,32,48,64]` 整桶执行;在线并发 14–30 落在桶间 → padding 行照算 attention/MoE;而 P11 离线门只测 b1/b8/b32(全是桶边界),结构性看不见 |
| P1 转正落地 | **取消** | 前置判定 FAIL,按冻结门不动默认 |
| P2 I30 v4 复核 | 未跑 | 旧树 `/root/moe-server-p6d` 已预部署 |

## 2. 已判决不要再碰(累计)

投机解码 NO-GO;I19 "融合换步时"否定(replay 有效已判决);整图 graph TP2 否决(I29);**分段图在线转正否决(P12:收益仅 b1,批量 ≥~8 净负)**;split=0 与 chunk-2048 默认化否决(P10);profiled placement 不采用;fused RMSNorm/容量收缩不采用;host-sync 移除与 mixed-split 已采纳;paged 已转正;tiny sdpa GB 门不适用(I27);双引擎 illegal access 按偶发记录不追。

## 3. 下轮 Top-N

> **本轮任务书 = [docs/PROMPT-P13.md](./PROMPT-P13.md)**(自包含:环境/命令/已判决清单/五项攻坚的假设与冻结门/红线/验收)。下表是它在 HANDOFF 里的摘要;冲突时以任务书为准。

| 优先 | 项 | 说明 |
|---|---|---|
| **0** | **I30 定案(同 session 新旧树 ABA)** | 不去追 0.343 绝对门(绑机器状态),改看相对差:现树 ≤ 旧树 → 关闭 I30。~1.2 h,2 次加载 |
| **1** | **I33 padding 税实测(b16 vs b17)** | 0.6B 零 30B 加载,~1 h。阶跃成立 → 图路径定性"单流专用",分支关闭 |
| **2** | **1k 长上下文端到端(I8/I26 从未复测)** | 必须同时报**拒绝率**,否则等于用 429 刷 SLO。~1.5 h,1 次加载 |
| **3** | **eager 生产栈容量上界** | 现有饱和数据只覆盖段图臂;eager 上界未知。可与 P2 同一次加载,~45 min |
| 4 | (可选大课题)KV 量化 I20 | 1–2 天新课题,先做 int8 可行性探针,**不建议本轮做** |
| 5 | 若要复活图:先做密桶/`cap/bs` 回退 | 先看复活门:**1.0 档须优于 eager ≥3%**,持平不值得 |

## 4. 本轮新 SOP/坑

1. **离线图门必须测非桶边界 batch**。桶集 `[1,2,4,8,16,32,48,64]` 整桶 replay,P11 只测 b1/b8/b32 = pad 比 1.0,于是 b9–b31 的 padding 税被系统性漏掉,直到在线矩阵才暴露。以后任何"按桶捕获"的改动,门里必须有 b12/b20/b24 之类的中间点。
2. **跨时段的 A/B 必须补同 session 复跑**。P12 的 A' 与 B' 相隔 19 h(含一次 ssh 失联),B2 复跑把 B' 复现到 0.1% 以内才让判定立住。以后两臂尽量相邻跑;不得不隔时,必须补一次复现臂。
3. **`EADDRINUSE` 是已知 flake 不是回归**。`llm_engine._get_distributed_init_method` 先 bind 临时端口再关 socket,torch 随后 rebind,存在极小竞态(`llm_engine.py:19-29`)。本轮 `p2_grouped_mm` 首次即撞上,单独重跑 rc=0。遇到先重跑一次,不要当代码回归。
4. **ssh 两种失败形态要分开**:`Exceeded MaxStartups` = 自己握手太密 → 等 60–90 s 再连、用 ControlMaster 复用;`Permission denied (publickey)` = 服务端认证状态坏 → 见下方旧 SOP。
5. (继承)api_server 默认 ≠ 生产栈:必须显式 `--decode-moe-backend bmm_dense`;段图上服务必须 `--max-num-seqs ≤ ~64`;detached 链跨会话存活,恢复连接先查产物目录再重跑;运行时切后端用 `runner.call`;pgrep 用 `^/usr/bin/python3|^python3|^[s]pawn_main`;30B token 门丢首 batch(I31)。

## 5. 红线

见 [AGENTS.md](../AGENTS.md);重申:`NANOVLLM_SEGMENT_GRAPH` 默认**关**(在线判定 FAIL,不翻);`enforce_eager` 生产默认**True**(不动);远端伪 `.git` 绝不提交;一次一个 GPU 任务;本地 commit,不 push 除非用户明确授权。

## 6. 证据索引(本地 `benchmarks/data/p12/`)

- `p12_online_armA2.json` / `.log`(臂 A' 生产栈,sha256 `a25f9b52…48d604e8`)、`p12_online_armB2.json`(段图复跑,sha256 `f966d103…178d683`)
- `p12_online_armB.json`(段图首测,sha256 `a3d8e177…31026d8`)、`p12_sat_armB.json`(饱和,sha256 `6087244b…7931cf`)
- `p12_ab_verdict.json`(A' vs B')、`p12_ab_verdict_samesession.json`(A' vs B2)
- `p12_online_armA_groupedmm_decode.*`(首轮口径错误,作废留证)、`p12_online_armA2_partial.log`(早先 5/12 点)
- 冻结门与全程记录:`benchmarks/P12-serving.md`;服务器侧 `/root/p12_api_arm{A2,B2}.log`
