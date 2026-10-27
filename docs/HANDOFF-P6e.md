# HANDOFF-P6e — 交接（2026-10-27 19:50）

> **本文是当前权威交接**，取代 [`HANDOFF-P6d.md`](./HANDOFF-P6d.md)（P6d 的 P0–P2 任务已全部执行完）。
> **下一轮排期**见 [`P8-24h-plan.md`](./P8-24h-plan.md)（其 P1 已按本轮结论改为"用 launch-bound 判定做 norm/rope 融合 + host-sync A/B"）。
> 规则不变：文档与代码/实测冲突时以代码和实测为准；每个实验先写门再跑；一次只跑一个 GPU 任务；默认本地 commit、不 push。

---

## 0. 30 秒状态

- **生产栈**：`attention=triton_paged`（本轮转正）、`moe_backend=grouped_mm`（prefill）、`decode_moe_backend=bmm_dense`、`enforce_eager=True`、`scheduler=stall_free`（在线）/`fcfs`（离线）、TP2/EP2。
- **代码**：`feat/musa-port`，HEAD `c552d84`（含 `a296069` 代码、`d5779b6` 文档），工作区干净（仅 `.DS_Store` 未跟踪），**未 push**。
- **服务器**：`ssh s4000b`，**双卡 0 MiB、无 python/spawn 进程、`/dev/shm` 无 `nanovllm*`、无 lock 与 watchdog**。会话结束于 19:50，没有在跑的任务。
- **并行会话提醒**：另一个会话在 `693e846` 提交了 P8 规划文档；本轮只做了增量更新，没有改写其内容。如果它继续跑 GPU，先让它读 `benchmarks/P6-layer-timing.md`，避免重复测量。

## 1. 本轮（P0–P2）结果

### P0 — I4/L5：✅ 通过并转正
- 四档（0.5/0.75/0.9/1.0，参数与 v3 完全一致）**512/512 成功、全部 100% SLO**。
- 100% 档：TTFT p95 **1.203 s**（v3 1.270）、TPOT p95 **0.327 s**（v3 0.401，**−18.4%**）、goodput **0.776**（v3 0.771）。
- 默认已改：`nanovllm/config.py:33` + `api_server.py --attention-backend` = `triton_paged`；用默认参数重启并冒烟验证。
- per-backend 基线 `benchmarks/p2-reference-8tok-paged.json` 已生成（8-token 输出与 sdpa 参考逐 token 相同）。

### P0 — "paged 启动挂死"：✅ 误判，实为静默慢加载
- `load_model` 的 FP8 逐 expert 反量化 **351.4 s（rank0）/354.0 s（rank1）**，warmup 16.4 s，KV 0.08 s，到 `/health` **377 s**；冷缓存 11–12 min。
- faulthandler 栈全部在 `loader.py:103/108`。**paged 不参与启动**（warmup 只跑 prefill，paged 首次 decode 才 import）。
- 运维：启动窗口 ≥20 min；判活用 `NANOVLLM_TRACE_INIT=1`。

### P1 — I19：✅ launch-bound
- 0.6B TP1 eager b1，两次重复：步时 **53.7–61.2 ms**，kernel 设备 **29.3–32.4 ms**，`device_busy` **0.478–0.603**（门 <0.70）。
- **2,401 launches/步**（35 种 kernel），**13.5 µs/次**；每层 host **1.84–2.13 ms** × 28 ≈ 整步。
- 成本是 RMSNorm/RoPE/mask 的逐元素小 kernel（top-15 占设备时间 74%），不是算力。

### P2 — I8：✅ 有效跑，负结果
- 1,018–1,154 token unique prompt，**384/384 完成**，rank1 全程存活（5 s watchdog）。
- 三档 SLO **0%**；TTFT p50 **3.84/8.28/10.22 s**；TPOT p50 0.418/1.175/1.419 s；并发 25/80/104。
- 50% 档 TTFT≤2 s 的请求 0/128、TPOT≤0.5 s 的 83/128；**空闲单请求 TTFT 0.368 s、TPOT 0.144 s**。
- 结论：瓶颈是"1k prefill + decode 混跑 + 无准入控制"的排队，不是单请求能力，不是故障。
- ⚠️ 长度门按字面未过：二/三档 p50=1154（门上限 1088），原因是 request-id 数字分词；实际 0.99–1.13× 目标，报告已注明。

### P1.5 — 投机解码再断言（0 GPU）
- 解析式用实测 α/c 标定：k=1 预测 0.727× vs 实测 0.725×（误差 0.3%）。
- 1.3× 门需 α≈0.80（k=1），实测 0.0056 → **差 142×**；完美草稿上限仅 1.45–2.21×（c=0.384 钉死）。
- I19 修好后门所需 α 降到 ~0.40。**S6 仍是唯一复活路径，且前置 I19。**

## 2. 现在确定的事实（不要再重测）

| 项 | 结论 |
|---|---|
| attention 后端 | paged 为默认，L1–L5 全过；sdpa 保留 opt-in |
| 启动耗时 | 377 s（热）/ 11–12 min（冷）；silent 窗口在 `load_model` |
| I19 | launch/CPU-bound，~2.4k launches/步 |
| 长上下文 | 可服务，但在 0.435 QPS 就 0% SLO（TTFT） |
| 投机解码 | NO-GO，且有解析上界；不重跑 |
| 图 | 仍冻结（I1/I2 0/5，`enforce_eager=True`） |

## 3. 下一步（按 ROI）

1. **I19 的 eager 收益**：融合 RMSNorm/RoPE（消掉 Cast/Pow/Rsqrt/Add/Mul 的 launch 洪水），逐项 A/B，token 全等 + 时间下降为门。
2. **I26 准入/背压 + I23 mixed-step 后端**：直接对应 I8 的 TTFT 崩溃；mixed step 目前走 prefill 路径，decode 侧失去 bmm_dense/paged。
3. **E1 paged autotune**：已解闸（I4 转正），但 ROI 低于 1/2。
4. **30B TP2 抽查 I19**：确认 MCCL 下仍 launch-bound（预期是，只会更差）。
5. **I20 KV 量化 / I21 per-rank KV 容量（730 vs 728）**：见 P8 文档，周级/半小时级。

## 4. 硬约束与 SOP（本轮修正过的）

- 一次一个 GPU 任务；30B 加载前清 shm、确认双卡 0 MiB；长任务带锁 runner + `setsid nohup` + 落盘日志。
- **pgrep 必须锚定**：rank1 的 spawn 子进程 argv 是 `/usr/bin/python3`，`^python3` **匹配不到**；用 `^/usr/bin/python3|^python3|^[s]pawn_main`。`pgrep -f "nanovllm...api_server"` 会匹配发起命令的 bash 自身（本轮曾把自己的 ssh shell 杀掉）。
- 启动窗口 ≥20 min（冷）；不要用"显存已占"推断"权重已加载完"。
- 不 push、不 reset、不在服务器伪 `.git` 提交；不改 `p5_decide.py` 阈值；不改 site-packages；保持 `enforce_eager=True`。
- 不要重复：投机解码、30B graph A/B、I4 诊断、已定案实验。

## 5. 证据索引（sha256 见各报告）

| 内容 | 位置 |
|---|---|
| I4/L5 v4 报告 | `benchmarks/P6-serving-v4.md`；`benchmarks/data/p6d/online-v4-paged.json`（`8c113f7b…`） |
| I19 逐层报告 | `benchmarks/P6-layer-timing.md`；`benchmarks/data/p6d/p6_layer_timing_06.json`（`ec7e0207…`） |
| I8 长上下文报告 | `benchmarks/P6-i8-long-context.md`；`benchmarks/data/p6d/p6_i8_long_unique_v2.json`（`1249e49b…`）、`i8_watch.log`（`dd3b0c1e…`） |
| 投机解码再断言 | `benchmarks/P7-spec-decode-addendum.md`；`benchmarks/data/p7/p7_spec_ceiling.json`（`d7dd9e07…`） |
| paged 参考基线 | `benchmarks/p2-reference-8tok-paged.json` |
| 服务器原始文件 | `/root/online-v4-paged.json`、`/root/p6_layer_timing_06.json`、`/root/p6_i8_long_unique_v2.json`、`/root/p6d_*.log` |

## 6. 台账快照

- ✅ 已关闭：I4、I8、I5、I6、I7、I9、I10、I11、I13、I15、S1。
- 🔴 未结：I1（图 0/5）、I19（launch-bound，已有下一步）。
- 🟡 未闭合：I2、I3、S2–S5（搁置）。
- 🟢 低优先：I12、I14、S6、E1（已解闸）、E2。
- ⚪ 厂商：E3–E6。
- P8 新增：I20–I26（见 `docs/ISSUES.md`）。
