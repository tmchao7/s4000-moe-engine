# HANDOFF — 当前交接(滚动更新;本版 = P9 轮收官,2026-10-28)

> 本文是**唯一交接文档**(由 HANDOFF-P10 重命名,历史版本在 git 历史)。接手先读本文,再按 [AGENTS.md](../AGENTS.md) 核验环境。执行日志(逐实验冻结门→判定)见 git 历史 `docs/NIGHTLY-LOG.md`。
> 规则不变:先写门再跑;一次一个 GPU 任务;负结果照录;本地 commit 不 push;到点无论进度必须停下写交接。

## 0. 30 秒状态

- **仓库**:`feat/musa-port`,P9 全部改动已本地 commit;工作区干净。
- **服务器**:双卡 0 MiB、锚定 pgrep 无进程、`/dev/shm` 无 `nanovllm*`、无本轮 lock(收官时已核验)。
- **生产栈(未变)**:`attention=triton_paged`、`moe_backend=grouped_mm`、`decode_moe_backend=bmm_dense`、`enforce_eager=True`、`scheduler=stall_free`、TP2/EP2、chunk-256。30B eager b1 decode 步时 142.06 ms(P8 基线沿用)。
- **P9 净代码变化**:① `tests/p2_grouped_mm.py` 钉 `attention_backend="sdpa"`(harness 修复);② 新增 `tests/p9_mccl_graph_probe.py`(I29 分级探针,含 watchdog/phase 标记);③ 新增 `tests/p9_placement_ab.py`(placement A/B + map 生成);④ `benchmarks/data/p6d/online-v4-paged.json` 补拷到服务器(v3 单测 15/15 需要)。(P9 曾加 `NANOVLLM_GRAPH_REPLAY_BARRIER` 实验 flag,已在文档/代码瘦身 commit 中删除——I29 workaround 已否定。)

## 1. P9 轮判决(证据在 `benchmarks/P9-*.md`)

| # | 项 | 判决 | 关键数字 |
|---|---|---|---|
| P1 | I29 图探针 | **定性收口**:TP2 引擎级 MUSAGraph+MCCL 间歇性 replay 楔死;纯张量(含引擎协议复刻六桶共享 pool)全绿、TP1 全绿;**最小复现 = 0.6B TP2 paged 首次 replay(3/3,2 min 加载)**;sdpa 撞 2 次 generate 后楔;**replay barrier 否定**;30B 双轨门维持结构性阻断 | `P9-graph.md` |
| P2 | profiled placement | **动机假设否定**:imbalance **0.1685→0.0009(−99.5%)** 但 decode b32 303.28→303.30 ms(**+0.01%**)→ rank 失衡不是 30B decode 瓶颈(与 I19 固定等待主导互证);**不采用为默认**,map 流程保留 | `P9-moe.md` |
| P3 | TTFT(chunk-2048) | **P8 未过的改进门达成**:50% 档 TTFT p50 3.31→**1.036 s(−69%,门 ≤2 s)**;75%/100% 档 −70%/−73%;50% 档 SLO std 0.8%→**100%**;高负载 TPOT 同时 −21~33%。**chunk-2048 = 1k 场景推荐配置** | `P9-long-context.md` |
| P4 | Poisson ×3 CI | 四档 ×3 **全 100% SLO(std+strict)**,Wilson CI [99.0%,100%],极差 ≤0.018 s;容量 = ≥99% SLO 最大达成 QPS **0.713**(未 bracketed);回归门:TTFT p95 过(1.19≤1.263)、**TPOT p95 0.387–0.397 vs 门 0.343 FAIL(+13~16%,稳定)→ 新开 I30** | `P9-serving.md` |
| P5 | 收尾 | sha256、ISSUES(I29 收口/I30 新开)、PLAN、本文、服务器清零 | `P9-nightly.md` |

## 2. 未完成 + 原因(不要当成"没试过")

| 项 | 状态 | 原因 |
|---|---|---|
| 8B TP2 paged 楔死确认 | 未跑(有意) | 0.6B 已是 3/3 最小复现,8B 预期同楔、边际信息量低;时间让给 P2–P4 |
| 30B 图 A/B | 维持否决 | I29 在 0.6B 即楔死,双轨门结构性阻断(红线:前置门不过禁止 30B 图 A/B) |
| 分段图(attention eager + linear/MoE 进图) | 未实现 | 周级工程;I29 复活的第一候选路径 |
| I30 mixed-split 在线 A/B | 未跑 | 需一发 30B 加载;P10 第一优先之一 |
| chunk-2048 短 prompt 矩阵 + 默认切换 | 未跑 | 需一发 30B 加载;与 I30 A/B 可合并同加载 |
| Poisson >1.0 饱和 bracketed | 未跑 | 四档全 100% 使容量上界未定;P8 v5 并发曲线(~2.14 req/s)可参照 |
| I28 最小 repro 存档 | 未做 | 低优先,沿用 P9 遗留 |

## 3. 下一轮 Top-5(按信息量排序)

1. **I30 判定 + chunk-2048 短 prompt 验证(合并一发加载)**:api_server 起 `NANOVLLM_MIXED_SPLIT=0` 跑 P4 同矩阵 → 判"混步税"真伪;再 `--prefill-chunk-size 2048` 跑同矩阵 → 判默认切换。两个决策都只需一次加载(重启 flag 即可,不用重载权重…… 若用 api_server env/flag 热切换则一次加载两实验;不行就两次启动,权重热缓存 ~6 min)。
2. **I29 分段图原型(若要做图)**:attention 走 eager、linear/MoE 进图(SGLang 式);先 0.6B TP2(本轮最小复现口径)验证不楔,再 8B、30B。或者等厂商 MUSAGraph+MCCL 修复,用 `tests/p9_mccl_graph_probe.py` 回归。
3. **Poisson 饱和 bracketed**:fractions 1.2/1.4(或 --max-qps 提高)找到 slo_standard ≥99% 的上界,补全容量曲线(×3 repeats)。
4. **TTFT 下一层**:chunk-2048 后 50% 档 TTFT p50 1.036 s,若想逼近空闲单请求 0.368 s,剩余空间在混步排队与 prefill kernel(profile 1k prefill 哪层贵——P9 未做的 kernel 级 profile)。
5. **E1 paged autotune / E2 fused_moe 配置**:仍低优,预算富余再做。

## 4. 本轮新 SOP(P9 血泪,继承 P8 全部)

1. **pgrep/pkill 自匹配**:远端 shell 命令行里含模式字符串时 `pgrep/pkill -f` 会匹配到执行命令的 shell 自身(本轮两次自杀:一次 pkill -f "p8_graph_padfix"、一次 spawn_main 模式)。**一律用 `[p]`/`[s]` 括号技巧或按 PID**。
2. **观测管道会吞掉崩溃现场**:`ssh ... | grep | tail` 的缓冲在连接被杀时丢弃 stderr——**长任务一律 nohup 全量日志落盘,再从日志 grep**;楔死排查前先确认"[graph]/[padfix] 进度行是否还在新增"。
3. **楔死三形态**:双卡 100% 对称自旋 / rank0 死+rank1 旋 / rank1 死+rank0 旋——都是 I29/I12 家族,处理统一:记录现场(mthreads-gmi + top + 最后日志行)→ 按 PID kill → rm shm → 双卡 0 MiB 再继续。
4. **MUSA launch watchdog 10 min 自杀会污染现场**:楔死后 rank 会收到 "the launch timed out and was terminated" 然后 DistBackendError teardown——**要在 watchdog 触发前(≤8 min)抓现场**;这同时说明"楔死 10 分钟"是可恢复的(设备会 reset),但 rank 状态已毁。
5. **placement 是加载期配置**:A/B 必须跨加载;跨加载 A/B 的口径 = 同一冻结工作负载 + 臂内 CV 门 + 解析式指标(如 imbalance 由 counts×ownership 计算)替代 interleaved。
6. **P0 单测环境**:本地 Mac 无 pytest/tqdm,三套单测都在服务器跑;`p2_grouped_mm.py`/`p6_bmm_dense_tiny.py` 用 `/tmp/tiny_moe`;v3 单测需要 `benchmarks/data/p6d/online-v4-paged.json` 在服务器 repo 内(已补)。
7. (继承)运行时切换走 `runner.call`;加载窗口只做测量;一次一个 GPU 任务;A/B 交错/中位数/CV≤5%;失败 init 先 destroy_process_group。

## 5. 红线(继承 + 本轮未变)

不 push;不 reset;远端 `.git` 伪仓库;不改 site-packages;不改 `p5_decide.py` 阈值;不用 `vllm_fused`/`mp22` 默认;不全局 `bmm_dense`;pgrep 锚定+括号技巧;一次一个 GPU 任务;陌生进程不杀只记录;长任务 nohup+lock+落盘;不留自动重启 waiter;30B 加载窗口 ≥20 min(冷);代码先行;不重跑已定案实验(P8 §1 + P9 §1)。

## 6. 证据索引(本地 `benchmarks/data/p9/`,sha256 见 `benchmarks/P9-nightly.md` §3)

P0:`p9_p0_regressions.log`、`p9_grouped_mm.log`;P1:`p9_mccl_graph_probe*.json`、`p9_mccl_probe3.json`、`p9_padfix_tiny_tp2.json`、`p9_06b_padfix.log`、`p9_06b_sdpa_outer.log`、`p9_barrier_exp.log`、`p9_barrier2.log`;P2:`p9_placement_{linear,profiled}.json`、`p9_expert_map.json`、`p9_place_*.log`;P3:`p9_long_context_c2048.json`、`p9_l3_*`;P4:`p9_serving_v5_ci.json`、`p9_l4_*`。

## 7. 如果只能做三件事

1. **I30 混步税 A/B + chunk-2048 短 prompt 验证**(一到两发加载,两个默认值决策,I19 之谜的又一切面)。
2. **Poisson 饱和 bracketed**(把容量曲线闭合,四档全 100% 说明上界还没到)。
3. **I29 分段图原型或等厂商**(图是 63→40 ms 的现成收益,但只走 attention-eager 路线,别再整体进图)。
