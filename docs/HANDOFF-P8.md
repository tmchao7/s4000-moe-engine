# HANDOFF-P8 — 24h 任务开跑前交接（2026-10-27 23:00）

> **状态:已被 [HANDOFF-P9.md](./HANDOFF-P9.md) 取代**(2026-10-28 收官)。本文仅作起点快照保留;当前状态以 P9 为准。

> **本文是 P8 执行轮次的起点交接**，写给即将接手 24h 的 Agent。上一轮交接是 [`HANDOFF-P6e.md`](./HANDOFF-P6e.md)（P0–P2 已完成）。
> 执行手册：[`P8-24h-plan.md`](./P8-24h-plan.md)。架构与差距：[`P8-architecture-review.md`](./P8-architecture-review.md)。复制给 Agent 的提示词：[`PROMPT-P8.md`](./PROMPT-P8.md)。
> 规则：每个实验先写门再跑；一次只跑一个 GPU 任务；失败/负结果照样记录；默认本地 commit、不 push。

---

## 0. 30 秒状态

- **仓库**：`feat/musa-port`，HEAD 以 `git log -1` 为准（起点在 `7d0b7b8` 之后），工作区干净（仅 `.DS_Store`）。**不需要再提交基线**。
- **服务器**：`ssh s4000b`，双卡 **0 MiB**、无 python/spawn 进程、`/dev/shm` 无 `nanovllm*`、无 watchdog/lock 残留。
- **生产栈**：`attention=triton_paged`（默认）、`moe_backend=grouped_mm`（prefill）、`decode_moe_backend=bmm_dense`、`enforce_eager=True`、`scheduler=stall_free`（在线）/`fcfs`（离线）、TP2/EP2。
- **没有在跑的任务**；24h 从 23:00 开始，到点（+24h）**无论是否完成都必须停、写文档、清服务器**。

## 1. 已经完成的事（不要重做）

| 项 | 结论 | 证据 |
|---|---|---|
| I4 / paged 转正 | L1–L5 全过；四档 512/512、100% SLO；100% 档 TPOT p95 0.327 s、TTFT 1.203 s、goodput 0.776 | `benchmarks/P6-serving-v4.md` |
| "paged 启动挂死" | **误判**：`load_model` FP8 反量化 351 s 无日志；到 `/health` 377 s（热），冷 11–12 min | `P6-serving-v4.md` §5 |
| I19 | **launch-bound**：2,401 launches/步、13.5 µs/次、device_busy 0.478–0.603、每层 host 1.84–2.13 ms × 28 ≈ 整步 | `benchmarks/P6-layer-timing.md` |
| I8 长上下文 | **有效跑、负结果**：1.0–1.15k token、384/384、rank1 存活；三档 SLO 0%；瓶颈 TTFT（3.84/8.28/10.22 s p50），空闲单请求 TTFT 0.368 s | `benchmarks/P6-i8-long-context.md` |
| 投机解码 | **关闭**：α 0.0056/0.0068、c 0.384/0.386、k=1 实测 0.725×；解析上界：1.3× 门需 α≈0.80，完美草稿仅 1.45–2.21× | `benchmarks/P7-phase0.md`、`P7-spec-decode-addendum.md` |
| per-rank KV 块数 | **已知 bug**：rank0 730 / rank1 728，高 block id 会越界；本轮 P0 修 | `P6-serving-v4.md` §4、架构书 §2.3 |
| 调度层 | 已判定只补三个硬缺口：KV 准入/429、mixed-step 走 decode 路径、`can_allocate == -1` 的队头阻塞 | 24h 计划 §4.5 |
| radix tree | **条件项，不排期**：当前是 256 粒度哈希链、无淘汰；radix 需小页（16/32）+ 节点 LRU + 改 kernel 页断言 | 24h 计划 §11 |

## 2. 已关闭、不得重开的方向

- **投机解码**（模型草稿 + 离线终裁）：不再排期、不重跑、不写终裁。复活条件：I19 解决（c≤0.1）+ 训练过的头（α≥0.40 @k=4）+ 产品需要降单流延迟。
- **I19 逐层计时**：已测；本轮只做融合与 sync 的 A/B。
- **I8 基线重测**：已测；本轮只在三个硬缺口修复后重测。
- **radix tree / 缓存淘汰 / 小页**：本轮不做（条件见上）。
- **30B graph A/B**：未过 8B 5/5 前禁止。
- **I4 诊断**：已定案，不重跑。

## 3. 本轮要交付的（A–F）

1. **A 提速**：至少一项确定性收益——fused RMSNorm（约 1,000/2,401 launches）优先，RoPE 次之，host-sync 便宜项并行；bmm_dense 容量收缩是 30B 的单点大收益。
2. **B 图**：padding-slot 根因判决（4 h 硬停）。三个验证门：slot 0 K/V 回读、共享前缀 token 全等、capture 前 paged JIT 预热。
3. **C MoE**：fold 前真实负载统计 → 容量收缩（M1）→ placement（M2）→ route/pack 微基准。
4. **D 长上下文**：三个硬缺口修复后重测，**改进门 = 1k 50% 档 TTFT p50 ≤2 s**。
5. **E 服务化 v5**：并发曲线 + Poisson 四档 + 3× 重复 CI；**双 SLO 并列**；p99 仅参考。
6. **F 收尾**：sha256、ISSUES/PLAN/HANDOFF 更新、本地 commit、服务器清零。

## 4. 第一个 30 分钟（照做）

```bash
# 1) 本地状态
cd /Users/tmchao7/repos/02_personal/moe-server-s4000 && git log -1 && git status --short
# 2) 服务器状态
ssh s4000b 'mthreads-gmi | grep -E "^[01] "; pgrep -fa "^/usr/bin/python3|^python3|^[s]pawn_main" || echo no_python; ls /dev/shm/nanovllm* 2>/dev/null || echo shm_clean'
# 3) 同步（本地为准；远端伪 git）
tar czf - <改动文件> | ssh s4000b 'tar xzf - -C /root/moe-server-s4000'
ssh s4000b 'cd /root/moe-server-s4000 && md5sum <改动文件>'
# 4) 读三份文档：HANDOFF-P8 → P8-24h-plan → P8-architecture-review
```

然后按计划 §4.1 做 **KV 块数一致化 + tiny 回归**，再进 benchmark v3 最小集。**不通过 tiny，不启动任何 30B 加载。**

## 5. 硬约束与 SOP（本轮踩过的坑）

- **一次一个 GPU 任务**；30B 加载前：按 PID kill → `rm -f /dev/shm/nanovllm*` → 双卡 0 MiB。
- **pgrep 必须锚定**：`^/usr/bin/python3|^python3|^[s]pawn_main`。rank1 的 spawn 子进程 argv 是 `/usr/bin/python3`，`^python3` 匹配不到；`pgrep -f "nanovllm...api_server"` 会匹配发起命令的 bash 自身（曾把自己的 ssh shell 杀掉）。
- **启动窗口 ≥20 min**（冷 11–12 min + warmup），判活用 `NANOVLLM_TRACE_INIT=1`；不要用"显存已占"推断加载完成。
- **代码先行**：改引擎代码的实验，代码 + 单测 + tiny 验证必须在加载 30B **之前**完成；加载窗口只做测量。代码没就绪就不要加载。
- **噪声协议**：0.6B 步时实测 53.7–61.2 ms（12% 波动）；A/B 必须交错 ≥5 次、取中位数、CV≤5%，否则判 INVALID。
- **图门来源**：`b1 ≥+15%` 定于 bmm_dense 之前；跑前写双轨判定（硬门 token 全等；≥15% 转正、+8~15% 记录待决策）。
- **并行会话**：仓库可能有另一个会话用同一台机器；发现陌生进程**不要杀**，记录并暂停；收尾只清自己记录过的 PID。
- **不 push、不 reset、不改 site-packages、不改 `p5_decide.py` 阈值**；不用 `vllm_fused` 默认、不用 mp22、不全局 `moe_backend=bmm_dense`。
- **不留自动重启的 waiter**；长任务 `setsid nohup` + lock + 落盘日志 + 短轮询。

## 6. 证据索引（本轮基线）

| 内容 | 位置 / sha256 |
|---|---|
| 在线 v4 | `benchmarks/data/p6d/online-v4-paged.json`（`8c113f7b…`） |
| I19 逐层 | `benchmarks/data/p6d/p6_layer_timing_06.json`（`ec7e0207…`） |
| I8 长上下文 | `benchmarks/data/p6d/p6_i8_long_unique_v2.json`（`1249e49b…`）、`i8_watch.log`（`dd3b0c1e…`） |
| 投机解码上界 | `benchmarks/data/p7/p7_spec_ceiling.json`（`d7dd9e07…`） |
| paged 基线 | `benchmarks/p2-reference-8tok-paged.json` |
| 原始日志 | `benchmarks/data/p6d/p6d_*.log`（不进 git） |

## 7. 如果只能做完三件事

按这个顺序保底，其余写"未完成 + 原因"：

1. **KV 块数修复 + tiny 回归**（正确性，5 分钟级，但不修会让长上下文结果不可信）。
2. **fused RMSNorm 的 A/B**（唯一不依赖图、不依赖 30B、且直接削减约 40% launch 的确定性收益）。
3. **三个调度硬缺口的代码 + 单测**（I8 的 TTFT 崩溃的直接解；即使来不及跑 L3，代码与单测也是可交接的资产）。

图的 4 h 与 MoE 的 L2 是"有则更好"；如果前 8 小时被加载/环境吃掉，宁可放弃图，也不要挤掉正确性修复。

## 8. 结束时必须写

即使 24 h 没做完，也必须：

1. `docs/HANDOFF-P9.md`（或 `HANDOFF-P8-followup.md`）：30 秒状态、已完成、未完成及原因、下一步、证据路径。
2. `docs/ISSUES.md`：本轮新增/更新的条目（I21–I26）。
3. 每份报告含 `schema_version`、`slo_definition`、`prompt_mode`、`warmup_done`、`percentile_method`。
4. 本地 commit（不 push）。
5. 服务器：双卡 0 MiB、无 python/spawn、`/dev/shm` 干净、无 lock/watchdog。
