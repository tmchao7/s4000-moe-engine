# HANDOFF — 当前交接(滚动更新;本版 = P11 轮收官,2026-10-29 凌晨)

> 本文是**唯一交接文档**(历史版本在 git 历史)。接手先读本文,再按 [AGENTS.md](../AGENTS.md) 核验环境与命令。规则不变:先写门再跑;一次一个 GPU 任务;负结果照录;本地 commit(不 push 除非用户授权);到点无论进度必须停下写交接、清服务器。

## 0. 30 秒状态

- **仓库**:`feat/musa-port`,P10 commit `0eb5eb0` 已在本地(P9 及之前已 push);P11 为本轮新 commit(本地,未 push——push 需用户明确授权)。
- **服务器**:已清(双卡 45.7/47.9 GiB free、无实验进程、无 nanovllm shm、旧栈临时目录已删、驱动脚本已删;`/root/p11_*` 证据文件仍在,收官后可 pull)。
- **生产栈(未变)**:`triton_paged` + `grouped_mm`/`bmm_dense` + `enforce_eager=True` + `stall_free` + TP2/EP2 + chunk-256 + `MIXED_SPLIT=1`。30B eager b1 decode 步时 142–159 ms(口径相关,见 §1)。
- **本轮净代码变化**:① `tests/p8_graph_padfix.py` 加 `--phase eager|graph|both` 双进程口径(绕开双引擎 illegal access)+ `--attempts`;② `tests/p11_step_ab.py`(0.6B 步时 A/B)、`tests/p11_30b_ab.py`(30B 双轨)、`tests/p11_30b_flip.py`(翻转率探针);③ `model_runner.py` 加 `NANOVLLM_SEG_DEBUG`(捕获打点,默认关)与 `NANOVLLM_SEG_SLOW`(捕获慢化,默认关)。全部默认惰性。

## 1. 本轮(P11)判决(证据 `benchmarks/P11-graph.md` / `P11-serving.md`)

| 项 | 判决 | 关键数字 |
|---|---|---|
| **分段图 S2 主判决(0.6B TP2 paged)** | **PASS** | 224 图/rank 零失败、GB 5/5 bitwise、GC/GA 过、零楔死;两进程口径绕开双引擎 bug |
| **异步抗性(生产模式)** | **PASS** | 无 LB 捕获首攻成功(0.6B GB 5/5 bitwise 异步复验);P10 异步偶发 3/3 未复现;LB 中途 toggle 无效(env 启动缓存) |
| **0.6B 步时 A/B** | **PASS,−47.2%** | eager 55.85 → segment 29.50 ms,CV≤0.32% |
| **8B 门** | **PASS** | 288 图零失败、GB 5/5 bitwise、异步零楔死 |
| **30B 双轨门(P9 冻结口径)** | **五项全 PASS** | 768 图/rank 零失败;**b1 152.85 → 116.63 ms = −23.7%(1.31×),超 ≥+15% 转正门**;b8 持平 +0.5%;b32 +3.8%(真实回退);显存 +0.00 GiB;跨臂逐位 0 翻转(b1/b8/b32) |
| **I19 固定等待(20–30 ms 之谜)** | **判决:b1 主体 = 逐算子 host launch 开销,replay 可回收** | b1 消 36.2 ms/步;与 P8 fused RMSNorm 负结果自洽(单算子融合只消一类 launch,replay 消掉几乎全部);残余 116.6 ms = device + 每层 2 次 eager AR + 驱动 |
| 双引擎 illegal access(P10 遗留) | **3/3 未复现,降级偶发记录** | 打点开关 `NANOVLLM_SEG_DEBUG` 保留 |
| I30 v4 口径复核 | **中止,判定未产出**(紧急收尾) | 旧栈(d5779b6)0.5 档实速 ~12 min/跑,只完成 0.5×2(SLO 100%,无 TPOT p95)→ 转下轮 |
| Poisson 饱和 bracketed | 未跑 | 转下轮,与在线 A/B 顺路 |

**分段图至此全梯子(tiny→0.6B→8B→30B)验证完成,已具转正资格**(S2/S3 门全过 = 红线必要条件满足)。本轮**未翻默认**:转正动作必须附带在线 L3 SLO A/B(b32 +3.8% 高载代价、分段只覆盖纯 decode 步/mixed 步仍 eager)。

## 2. 已判决不要再碰(累计)

投机解码 NO-GO;I19 "融合换步时"因果链否定(单算子融合;replay 已判决有效);整图 graph TP2 否决(I29,barrier 无效);split=0 与 chunk-2048 默认化否决(P10);profiled placement 不采用;fused RMSNorm/容量收缩不采用;host-sync 移除与 mixed-split 已采纳;paged 已转正;**tiny sdpa GB 门不适用(I27)**;双引擎 illegal access 按偶发记录不追(无复现不修)。

## 3. 下轮 Top-N(长城任务;按收益排序)

| 优先 | 项 | 说明 |
|---|---|---|
| **1** | **segment 在线 L3 A/B + 转正决策**(最大收益在手) | 现行栈 vs `NANOVLLM_SEGMENT_GRAPH=1`,v4/P9 同矩阵(`--mode poisson --max-qps 0.87 --fractions 0.5,0.75,0.9,1.0 --requests 128 --prompt-words 128 --repeats 3`),两臂顺序各一轮服务(启动 ~7 min + 矩阵 ~40 min/臂)。门:SLO 双门不劣化 + 1.0 档 TPOT p95 不劣于 eager 臂 + 低载档 TTFT/TPOT 改善可见;顺路跑 `--fractions 1.2,1.4 --repeats 3` 补饱和 bracketed。**过门 → 翻默认**(实现:把 flag 提升为 config 默认或 api_server 默认 env;翻后跑 4 单测套件 + 0.6B S2 快速复验)。注意 I31:在线无影响,但凡 token 级比对须丢首 batch |
| 2 | **I30 v4 复核重跑** | pre-P8 树重放(判定树冻结于 P11-serving.md)。**预算按旧栈实速**:0.5 档 ~12 min/跑;省时口径 = 只跑 0.9/1.0 两档 ×3 ≈ 50 min;全矩阵 ×3 ≈ 2h+。旧树 = `git archive d5779b6` 到 `/root/moe-server-p6d`(已验证可跑;api_server 默认即 v4 栈) |
| 3 | b32 +3.8% 回退归因/缓解(可选) | 高载下每层 2 次 eager AR 的 host 开销占比上升;候选:AR 与下层 replay 重叠(缓冲已双份 h_attn/ping)、或高载自动降级 eager(bs 阈值)。收益上限 ~3.8% |
| 4 | 低优 | mixed-step segment 化预研(decode 行现在 mixed 步走 eager,扩大收益面);I20 KV 量化(周级);E1 autotune;E2 fused_moe |

## 4. 本轮新 SOP/坑

1. **两进程口径**:凡"同进程第二个引擎"相关实验,用 `p8_graph_padfix --phase eager|graph --eager-from`(token 本地比对),别再踩双引擎上下文。
2. **30B token 门必丢首 batch**(I31):进程首个 batch 与稳态逐位不同(一次性 warmup 态跃迁,arm 对称);稳态 greedy 逐位确定。参照 `tests/p11_30b_flip.py`。
3. **`MUSA_LAUNCH_BLOCKING` 进程中途 toggle 无效**(env 启动时缓存)——异步偶发只能靠重试/慢化/spawn 前设 env。
4. **ssh 单命令里别把 heredoc 和 tar 管道喂同一个 stdin**(本轮把驱动脚本写成空文件、tar 静默失败);分开两条 ssh。
5. **旧栈比现行栈慢得多**(0.5 档 ~12 min/跑):历史栈复现实验按实速 ×2 预算。
6. **b1 步时口径**:142.06(P8 nightly)/152.85(p11 drive_batch)/116.63(segment)——跨文档比较先对口径。
7. (继承全部 P8–P10 SOP:锚定 pgrep 括号技巧、nohup 全量日志、watchdog 抓现场、按 PID kill、失败 init 泄漏 PG 需 destroy、[p] 自匹配等)

## 5. 红线

见 [AGENTS.md](../AGENTS.md)(全部集中处);重申:`NANOVLLM_SEGMENT_GRAPH` **默认关**——S2/S3 门已全过,**允许改默认,但必须先过在线 L3 SLO A/B**(下轮 Top-1);生产 `enforce_eager=True` 在 A/B 过门并翻默认之前不动;不 push 除非用户授权;远端伪 .git 绝不提交。

## 6. 证据索引(本地 `benchmarks/data/p11/`)

`p11_s2_{eager,graph}.{json,log}`(S2 主判决)、`p11_s4_06b_async.{json,log}`+`p11_s4_tiny_*`(异步门)、`p11_p2_*`(双引擎未复现)、`p11_step_{eager,segment}_{1,2,3}.json`(0.6B 步时)、`p11_p4_8b_*`(8B 门)、`p11_30b_{eager,segment}_{1,2,3}.json`+`p11_30b_flip{,32}_*`(30B 双轨+翻转)、`p11_i30_p6d_client.log`(I30 部分)。冻结门与判定:`benchmarks/P11-graph.md`、`benchmarks/P11-serving.md`。
