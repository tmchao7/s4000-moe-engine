# PROMPT-P8 — 给执行 Agent 的 24 小时任务提示词

> 用法：整段复制（从下一行开始）发给执行 Agent。它需要能 ssh 到 S4000 服务器并有仓库写权限。
> 本提示词自包含；不要依赖对话上下文。计划书、架构书、交接文档都在仓库里，按 §0 读。

---

你接手一个 MUSA（摩尔线程 S4000 ×2）上的 Qwen3-30B-A3B MoE 推理引擎项目。仓库 `/Users/tmchao7/repos/02_personal/moe-server-s4000`，分支 `feat/musa-port`。服务器 `ssh -p 28036 root@connect.sha1.seetacloud.com`（别名 `s4000b`），代码目录 `/root/moe-server-s4000`（**伪 `.git`，不要用远端 git；以本地为准，用 tar/scp 同步**）。

你的任务：从 23:00 起执行一个 **24 小时**实验轮次。不要问问题；按计划自主推进；失败/负结果原样记录；不要伪造成功。到 24 h 结束（即使没做完）**必须停下并写交接文档**。

## 0. 必读（30 分钟，按顺序）

1. `docs/HANDOFF-P8.md` —— 起点状态、已完成的、不要再做的、SOP 修正。
2. `docs/P8-24h-plan.md` —— **执行手册**：时间表、加载账本、每个 Track 的冻结门、降级路线、§10 已删除项、§11 探索方向。
3. `docs/P8-architecture-review.md` —— 架构与差距；重点 §2.3（调度缺陷）、§2.7（图根因假设）、§3（MoE）、§4（I19 与融合清单）。
4. `docs/P8-moe-fusion-and-balance.md` —— §2（折叠路由）、§3（bmm_dense 容量收缩，最高 ROI）。
5. `docs/P8-benchmark-spec.md` —— 指标 v3 定义与 schema；注意 §10 的评审补充。
6. `benchmarks/P6-serving-v4.md` —— 当前在线基线（paged 已转正）。
7. `benchmarks/P6-layer-timing.md`、`benchmarks/P6-i8-long-context.md` —— I19 与长上下文的最新实测。
8. `docs/ISSUES.md`、`docs/PLAN.md` —— 台账；I4/I8 已关闭，I19 已定性。

## 1. 起点（已核实，不要重复确认）

- HEAD 以 `git log -1` 为准（基线在 `7d0b7b8` 之后）；工作区应干净（仅 `.DS_Store` 未跟踪）。**不需要再提交基线**。
- 生产栈：`attention=triton_paged`（默认）、`moe_backend=grouped_mm`（prefill）、`decode_moe_backend=bmm_dense`、`enforce_eager=True`、`scheduler=stall_free`、TP2/EP2。
- 30B 加载到 `/health`：热 ≈377 s（FP8 反量化 351 s），冷 11–12 min；收尾 teardown 5 min。**不要用"显存已占"推断加载完成**，用 `NANOVLLM_TRACE_INIT=1` 的 `[init]` 行判活。
- 已知事实：I19 = launch-bound（2,401 launches/步、13.5 µs/次、device_busy 0.478–0.603）；I8 长上下文三档 SLO 0%（瓶颈 TTFT，空闲单请求 TTFT 0.368 s）；投机解码已关闭（含离线上界）。
- 服务器当前应双卡 0 MiB、无 python/spawn、`/dev/shm` 无 `nanovllm*`。

## 2. 执行顺序（严格按序；时间盒见计划 §3）

### P0（H0–1.5，无 30B）
- 核验 HEAD/工作区/服务器；**先修 per-rank KV 块数不一致（730 vs 728）**：`allocate_kv_cache` 后广播 `min(num_blocks)`，门 = 两 rank `[kvcache] blocks=` 相同 + 高 block id 压测。不修则禁止长上下文 Track。
- tiny 回归：`tests/p2_grouped_mm.py --tp 1`、`tests/p6_bmm_dense_tiny.py`，六配置逐 token 全等、rc=0。
- 实现 benchmark v3 **最小集**：queue p50/p95、`goodput_tokens_per_s`、双 SLO（`slo_standard` + `slo_strict`，后者用 max ITL ≤1 s，**不用 ITL p95**）、`--request-timeout-s`（超时必须 `engine.cancel`）、`reject_429` 桶、p99 标 `p99_reference_only`。`--mode concurrency` / `--repeats` 可在 H15 前完成。
- 门：`tests/test_benchmarking_v3.py` 全绿；旧 v4 JSON 可回放。

### P1（H1.5–4，0.6B）融合优先
- **fused RMSNorm(add)**：bf16 进出、fp32 内累加，一条 Triton kernel 替换 `layernorm.py:24-38` 的 ~10 条逐元素 op；这是 2,401 次 launch 里约 1,000 次的来源。
- **fused RoPE** 可选，H3.5 未绿即顺延，不挤图与 MoE。
- host-sync 修复：prefill 的 96 次 `.tolist()` 上提；sampler 的 2 次 `bool()` 改为 host 侧 `all_greedy`。
- 门：token/L1-L2 门通过；0.6B b1 **−≥8%**；**交错重复 ≥5 次、CV≤5%**，否则判 INVALID（单次差值不可信，实测噪声 12%）。
- 证据：`/root/p8_fusion.json`、`/root/p8_sync_fix.json`、`benchmarks/P8-i19.md`。

### P2（H4–8）图 replay 根因（4 h 硬停）
- 假设：replay 把 padding 行 `slot_mapping` 填 0（`model_runner.py:563-571`），整桶执行导致每次 replay 覆写 KV slot 0。
- 先 `tests/p6_graph_min.py --generates 3` 复现；**捕获失败与 replay 漂移分开计数**。
- 修复：预留 scratch block（不交给 BlockManager），padding 行指向它；每次 replay 清 `block_tables[bs:]`。
- **三个验证门**：① replay 3 次后 slot 0 的 K/V 逐元素相等；② 共享 ≥256 token 前缀的两条请求 token 全等；③ capture 前确认 paged 已 JIT 预热（decode 上下文跑过一次）。
- 顺序：tiny 5/5 → 8B 5/5 → **才允许** 30B A/B（L1）。**跑前做门来源审计**：硬门 token 全等；性能门双轨（b1 ≥+15% 转正；+8~15% 记录待决策；<8% 判不划算）。
- 4 h 到点无论结果都停；失败写"否决 + 证据"。

### P3（H8–12，L2）MoE：容量收缩 → 放置 → 融合
- **代码必须在 L2 加载前完成并在 tiny 上单测**（加载窗口只做测量）。
- 真实负载统计：32 条中英 chat prompt，prefill 512 / decode 64；**必须在 fold 之前用 top-k ids 计数**（`local_pairs` 不可用），给出每层 counts、`S_max`、expert0 折叠比例、waste、rank imbalance。
- **第一优先：bmm_dense 容量收缩**（owned-only + 动态 `S=counts.max()`）。数值门：自确定 3/3 + L1/L2 软门（**不要求与现有 bmm_dense 逐 token 全等**）；性能门：MoE 段 −≥20%、步时 −≥10%。
- 然后 `linear` vs `profiled`（容量收缩开启；门 +5% 或 imbalance −10%）；`fused_grouped_mm` A/B；route/pack 微基准（只出设计）。
- 报告 `benchmarks/P8-moe.md`。

### P4（H12–16，L3）三个调度硬缺口 + 长上下文重测
- 代码 + 单测必须在 L3 加载前完成：
  1. **KV 准入/429**：超 `max_model_len` 返 400；容量不足不 assert、请求继续排队；超限返 429 并计数。
  2. **mixed-step 走 decode 路径**：batch 拆 prefill/decode 子批（或按 seq 打标签），decode 行用 `prepare_decode` + paged + bmm_dense。
  3. **队头阻塞**：`scheduler.py:181-184` 的 `can_allocate == -1 → break` 改为跳过候选继续尝试。
- 单测门：长 prompt 不可准入时短 prompt 仍被调度；429 计数正确；mixed decode 行 token 与纯 decode 全等。
- 重测：1k token、fractions 0.5/0.75/1.0、128 请求、64 输出；`prefill-chunk-size 256` vs `2048`；2k 单档用 **64 并发**（128×2k 会 preempt）。shared-prefix 长 prompt 作为第二负载（只报告）。
- 有效门：p50 ∈[960,1088]/[1920,2176]、rank1 存活、≥125/128 `[DONE]`。**改进门：1k 50% 档 TTFT p50 ≤2 s**（上轮 3.84 s），不达则 I26/I23 判未解决。

### P5（H16–19，L4）服务化 v3
- 固定并发 1/4/8/16/32/64；Poisson 四档（与 v4 同参）+ 100% 档 3 次重复。
- **双 SLO 并列**；p99 仅参考（128 样本）；错误分桶 `{none,timeout,http,reject_429,stream,token_count}`；warmup 完成后才开始计时。
- 容量 = 重复中 `slo_standard ≥99%` 的最大达成 QPS + Wilson 下界；回归门：0.87 档 SLO 100%、TPOT p95 ≤0.343 s、TTFT p95 ≤1.263 s。
- 报告 `benchmarks/P8-serving-v5.md`。

### P6（H19–24）收口 + buffer
- `benchmarks/P8-nightly.md`：总表 + 每份证据 sha256。
- 更新 `docs/ISSUES.md`（I21/I22/I23/I24/I25/I26 的本轮结论）、`docs/PLAN.md`、写 `docs/HANDOFF-P8.md`。
- **投机解码不排期、不重跑、不写终裁**（见计划 §10）。
- 本地 commit（**不 push**）；服务器清零（按 PID kill → `rm -f /dev/shm/nanovllm*` → 双卡 0 MiB）。
- H19–20 与 H22–24 是 buffer：只补测/修报告，不启新实验。**到点必须停并写交接**。

## 3. 红线（违反会毁掉一轮）

- **不 `git push`**；不 reset；远端 `/root/moe-server-s4000/.git` 是伪仓库，绝不在其上提交。
- 不改 `site-packages`；不改 `scripts/p5_decide.py` 任何阈值；新门用新脚本。
- 不用 `vllm_fused` 当默认；不用 `mp22`；不全局设 `moe_backend=bmm_dense`（prefill OOM）。
- **pgrep 必须锚定**：`pgrep -fa "^/usr/bin/python3|^python3|^[s]pawn_main"`。rank1 的 spawn 子进程 argv 是 `/usr/bin/python3`，`^python3` 匹配不到；`pgrep -f "nanovllm...api_server"` 会匹配发起命令的 bash 自身（曾把自己的 ssh shell 杀掉）。kill 前先列 PID。
- 一次只跑一个 GPU 任务；30B 加载前清 shm、双卡 0 MiB、无残留；发现**陌生进程不要杀**，记录并暂停。
- 长任务用 `setsid nohup` + lock + 落盘日志 + 短轮询；**不留自动重启的 waiter**。
- 不拿 tiny 的 token 一致性当图门；8B 修复不直接推 30B。
- **不要重跑已定案实验**：投机解码、30B graph A/B（未过 8B 前）、I4 诊断、I19 测量。
- 文档与代码/实测冲突时以代码和实测为准，先修文档再继续。

## 4. 交付格式

每个实验在 `docs/NIGHTLY-LOG.md` 追加：假设 → 唯一变量 → 预期（数值）→ **跑前冻结门** → 实际 → 判定 PASS/FAIL/INVALID → 证据路径 + sha256 → 决策。
每份 benchmark 报告必须写 `schema_version`、`slo_definition`、`prompt_mode`、`warmup_done`、`percentile_method` 五个字段。

## 5. 验收（A–F，缺一不可）

- **A 提速**：至少一项（fused RMSNorm/RoPE、host-sync、容量收缩）经交错 A/B：token/数值门通过且目标指标 −≥8%（CV≤5%）。
- **B 图**：padding-slot 修复后 tiny/8B 5/5 + 30B 按双轨门判决；或给出"不是这个根因"的证据并停止。
- **C MoE**：fold 前真实负载统计 + 容量收缩判决 + placement A/B + route/pack 微基准，每项有结论。
- **D 长上下文**：三个硬缺口修复后 1k 重测且 50% 档 TTFT p50 ≤2 s；2k 单档有效；rank1 全程存活。
- **E 服务化**：并发曲线 + Poisson 四档 + 100% 档 3 次 CI；双 SLO；容量结论。
- **F 收尾**：证据 sha256、ISSUES/PLAN/HANDOFF 更新、本地 commit（不 push）、双卡 0 MiB、无残留、shm 干净。

## 6. 失败处理

- 单问题 45 min：写"已排除 / 未排除 / 下一步"，切下一条。
- 30B 加载失败：原样记录，最多重试 1 次；仍失败跳过该 Track，转 0.6B/8B 或离线文档。
- 图 4 h 到点：停，写否决，转 MoE。
- A/B 的 CV >5%：判 INVALID，增至 10 次交错或放弃该 kernel。
- 代码没在加载前就绪：**不要加载**，用 buffer 补；加载后才发现要改代码，判该 Track 失败并记录。
- 任何"数字看起来不对"：先查口径（时钟域、分母、prompt tokens、是否混步），再下结论。

开始前先输出一行：`[P8] start — HEAD <sha>, server <0 MiB/有任务>, 24h from <time>`。然后按 P0 执行。
