# PROMPT-P8 — 给执行 Agent 的一昼夜任务提示词

> 用法：整段复制（从下一行开始）发给执行 Agent。它需要能 ssh 到 S4000 服务器并有仓库写权限。
> 本提示词自包含；不要依赖对话上下文。

---

你接手一个 MUSA（摩尔线程 S4000 ×2）上的 Qwen3-30B-A3B MoE 推理引擎项目。仓库在本地 `/Users/tmchao7/repos/02_personal/moe-server-s4000`，分支 `feat/musa-port`。服务器 `ssh -p 28036 root@connect.sha1.seetacloud.com`（别名 `s4000b`），代码目录 `/root/moe-server-s4000`（**伪 `.git`，不要用远端 git，以本地为准 scp/rsync 同步**）。

你的任务：按 `docs/P8-24h-plan.md` 执行一个 **24 小时**的实验轮次。不要问问题，按计划自主推进；遇到失败原样记录并降级，不要伪造成功。

## 0. 必读（按顺序，30 分钟）

1. `docs/P8-24h-plan.md` —— 时间表、每 Track 的冻结门、加载账本、降级路线。**这是你的执行手册**。
2. `docs/P8-architecture-review.md` —— 现状、差距、设计判断。重点读 §1.3（hidden 成本）、§2.7（图根因假设）、§3（MoE）、§4（I19）。
3. `docs/P8-moe-fusion-and-balance.md` —— 特别是 §2（折叠路由的 50% 无效行 / 65:1 倾斜）与 §3（bmm_dense 容量收缩）。
4. `docs/P8-benchmark-spec.md` —— 指标 v3 定义与 schema。
5. `benchmarks/P6-serving-v4.md` —— 当前基线（paged 已转正）。
6. `benchmarks/P7-phase0.md`、`benchmarks/P7-spec-decode-addendum.md` —— 投机解码已判 NO-GO，不要重跑。
7. `docs/ISSUES.md`、`docs/PLAN.md` —— 历史问题与门。注意：I4 的 L5 已在 v4 通过，文档中"I4 未完成/默认 sdpa"的表述已过时，以代码和 v4 报告为准。

## 1. 起点（已核实，不要重复确认）

- HEAD `fa0b113`；工作区有未提交的 v4 相关改动（`config.py` 默认 `triton_paged`、`model_runner.py` 的 `NANOVLLM_TRACE_INIT`、`api_server.py` 的 `--max-model-len`/faulthandler、`benchmark_online.py` 的 `--prompt-tokens`/`slo_success_rate`），以及未跟踪的 v4 报告、P7 附页、`tests/p6_layer_timing.py`、`scripts/p7_spec_ceiling.py`、`scripts/p2_write_reference.py`。**先把它们提交成基线**（不 push）。
- 生产栈：`attention=triton_paged`、`moe_backend=grouped_mm`（prefill）、`decode_moe_backend=bmm_dense`、`enforce_eager=True`、`scheduler=stall_free`、TP2/EP2。
- 30B 热加载到 `/health` 约 6 分 17 秒（其中 FP8 反量化 351 s）；冷缓存 13–15 min。**不要再用"显存已占"推断"加载完成"**，用 `NANOVLLM_TRACE_INIT=1` 的 `[init]` 行判活。
- 验收基线：在线 v4 四档 100% SLO、100% 档 TPOT p95 0.327 s / TTFT p95 1.203 s / goodput 0.776；离线 b32 110.6 tok/s（sdpa+bmm）/ b64 141.48（sdpa+bmm，paged+bmm 尚未测）。

## 2. 执行顺序（严格按序；时间盒见计划）

### P0（H0–1.5）基线 + 指标 v3
- 提交基线；同步服务器；跑 tiny 回归（`tests/p2_grouped_mm.py`、`tests/p6_bmm_dense_tiny.py`），六配置逐 token 全等才继续。
- 按 `docs/P8-benchmark-spec.md` §5 实现最小集：queue/prefill/decode 三段、`goodput_tokens_per_s`、`slo_strict`（含 `max ITL ≤1 s`）、`--request-timeout-s`、`--repeats`、`--mode concurrency`、`--prompt-mode mixed`、prefix hit rate、GPU 采样、报告出 p99。
- 新建 `tests/test_benchmarking_v3.py`（Wilson、token goodput、ITL SLO、error 分桶、linear percentile、schema 回放），全绿。

### P1（H1.5–4）I19 定性 + host sync
- 跑 `tests/p6_layer_timing.py`（0.6B、TP1），按脚本内冻结阈值判 `launch_bound / kernel_bound / mixed`。
- 实现两项修复并用 A/B 验证：① `_forward_prefill` 的 `cu_seqlens_q/k.tolist()` 上提到 `prepare_*`（当前每层 2 次，48 层 = 96 次/步）；② `Sampler` 去掉每步 2 次 `bool()`，在 host 由 `temperatures` 列表算 `all_greedy`。
- 门：token 全等；0.6B b1 步时中位数 −≥8%；prefill 1k step wall −≥10%；b32 sample −≥50%（30B 在 L2 验证）。不过门就保留 flag 默认关闭，写数字。

### P2（H4–8）图 replay 根因（4 h 硬停）
- **假设**：replay 把 padding 行 `slot_mapping` 填 0（`model_runner.py:563-571`），整桶执行导致每次 replay 覆写 KV slot 0；block 0 常是活跃请求首块 → 首个 replay 漂移。
- 先 tiny `p6_graph_min.py --generates 3` 复现（3/3 漂移）。然后实现 scratch slot（KV 多留 1 块，不交给 BlockManager；padding 行指向它；每次 replay 清 `block_tables[bs:]`）。
- 顺序：tiny 5/5 → 8B 5/5 → **才允许** 30B A/B（L1）。
- 冻结门（不可改）：图与 eager 逐 token 全等（b1/b8/b32）；b1 ≥+15%；b8 ≥+5%；b32/b64 ≥0.98；CV ≤5%；图显存增量 ≤2 GiB。
- 4 h 到点无论结果都停；失败写"否决 + 证据"，不做第二次。

### P3（H8–12）MoE（L2）
- 先测真实负载（32 条中英 chat prompt，prefill 512 / decode 64，`collect_expert_stats=True`）：每层 counts、`S_max`、expert0 折叠比例、waste ratio、折叠修正后的 rank imbalance。
- `linear` vs `profiled`（真实负载生成的 LPT map）：b32 decode ×3、b32 prefill ×3。门：吞吐 ≥+5% 或 imbalance −≥10%。
- **最高优先实验**：`bmm_dense` 容量收缩（只打包 owned slot，`S = counts.max()`）。先看 M0 的 `S_max` 是否显著小于 `T×K`；实现原型，测 b32/b64 的 MoE 每层时间与步时。门：逐 token 全等（首选）/ L1–L2 + 确定性；MoE 层 −≥20%；步时 −≥10%。
- 其次：`fused_grouped_mm` A/B（门：MoE −≥5%，数值门）；route/pack 微基准（门：−≥30%，24h 只到微基准 + 设计，不改主路径，除非时间富余）。
- 测 mixed step 与纯 decode 的 forward 差，给出在线 TPOT 影响上界。
- 报告 `benchmarks/P8-moe.md`，每项一句结论 + 数字 + 证据。

### P4（H12–16）长上下文 + 调度硬化（L3）
- 无 GPU 段：`add_request` 校验 `prompt_len + max_tokens ≤ max_model_len`；KV 不足返回 429 不 assert；客户端 timeout；runner watchdog 每 5 s 记录 `mthreads-gmi` 与 `[s]pawn_main` PID。
- 长上下文：`--prompt-mode unique --prompt-tokens 1024 --max-model-len 4096`，fractions 0.5/0.75/1.0，128 请求，64 输出；`--prefill-chunk-size 256` vs `2048` 两次（唯一变量）。2k 档单档观测。
- 有效门：prompt tokens p50 在 [960,1088] / [1920,2176]；rank1 全程存活；≥125/128 `[DONE]`。不满足记无效跑，最多重试 1 次。
- 报告 `benchmarks/P8-long-context.md`（含 preemption、KV 峰值、chunked 对比）。

### P5（H16–20）服务化 v3（L4）
- 固定并发曲线：`--mode concurrency --concurrency 1,4,8,16,32,64`。
- Poisson 四档（与 v4 同参）+ 100% 档 **3 次重复**。
- 容量定义（新）：3 次重复中 `slo_standard ≥99%` 的最大达成 QPS，报 Wilson 95% 下界。
- 回归门：0.87 档 SLO 100%，TPOT p95 ≤0.343 s，TTFT p95 ≤1.263 s；否则先解释口径差异。
- 报告 `benchmarks/P8-serving-v5.md`。

### P6（H20–24）收口
- `benchmarks/P8-nightly.md`：总表 + 每份证据 sha256。
- 投机解码终裁：维持 NO-GO，写死三条复活条件（I19 解决、α≥0.40 实测、EAGLE3/prompt-lookup 权重就绪）。不重跑 Phase 0。
- 更新 `docs/ISSUES.md`（I4 关闭；新增 I21 per-rank KV 容量不一致、I22 graph padding 写 slot 0、I23 mixed step 后端、I24 host sync、I25 benchmark v3、I26 KV admission/backpressure）、`docs/PLAN.md`、`docs/HANDOFF-P8.md`。
- 本地 commit（**不 push**）；服务器清零（按 PID kill → `rm -f /dev/shm/nanovllm*` → 双卡 0 MiB）。

## 3. 红线（违反会毁掉一轮）

- **不要 `git push`**；不要 reset 工作区；远端 `/root/moe-server-s4000/.git` 是伪仓库，绝不在其上提交。
- **不要改 `site-packages`**；厂商适配只在 `nanovllm/kernels/vendor_moe.py` 的内存里做。
- **不要改 `scripts/p5_decide.py` 任何阈值**（门先冻结再测；新门用新脚本）。
- **不要用 `vllm_fused` 当默认**（decode 慢 2.3–6.3×）；**不要用 mp22**（数值错误）；**不要全局 `moe_backend=bmm_dense`**（prefill OOM，只能 `decode_moe_backend`）。
- **不要用未锚定的 `pkill -f` / `pgrep -f`**（会杀掉执行命令的 bash/ssh 自身）——用 `^python3`、`^/bin/bash`、`[s]pawn_main`，或先列 PID 再按 PID kill。
- **一次只跑一个 GPU 任务**；30B 加载前必须清 shm、双卡 0 MiB、无残留进程。
- **不要留自动重启的 waiter**；长任务用 `setsid nohup` + lock + 短轮询。
- **不要拿 tiny 的 token 一致性当图门**；8B 修复不直接推 30B。
- **不要为了"有产出"重跑已定案实验**（I1/I2/I4 的结论已有证据；P7 已判 no-go）。
- 文档与代码/实测冲突时，**以代码和实测为准，先修文档再继续**。

## 4. 交付格式

每份报告必须包含：唯一变量、**跑前写下的冻结门**、实际值、判定（PASS/FAIL/INVALID）、失败/负结果的原始数字、证据路径 + sha256。
每个实验在 `docs/NIGHTLY-LOG.md` 追加一段（假设 → 唯一变量 → 预期 → 门 → 实际 → 决策）。

## 5. 验收（A–F，缺一不可）

- **A** I19：`device_busy` 判定 + 至少一项 sync 修复经 A/B 验证（token 全等 + 时间下降）。
- **B** 图：根因修复后 tiny/8B 5/5 + 30B 过冻结门；或给出"不是这个根因"的证据并停止。
- **C** MoE：真实负载统计 + placement A/B + 容量收缩/融合微基准，每项有判决。
- **D** 长上下文：1k 与 2k 各一档有效跑（或无效 + 根因）。
- **E** 服务化：v3 口径的并发曲线 + Poisson 四档 + 100% 档 3 次 CI；容量结论。
- **F** 收尾：证据 sha256、文档更新、本地 commit（不 push）、双卡 0 MiB、无残留进程、`/dev/shm` 干净。

## 6. 失败处理

- 单问题 45 min 写"已排除 / 未排除 / 下一步"，切下一条。
- 30B 加载失败：原样记录，最多重试 1 次；仍失败跳过该 Track，转 0.6B/8B 或离线文档。
- 图 4 h 到点：停，写否决，转 MoE。
- 任何"数字看起来不对"：先查口径（时钟域、分母、prompt tokens、是否混步），再下结论。

开始前先输出一行：`[P8] start — HEAD <sha>, server <0 MiB/有任务>, plan 24h`。然后按 P0 执行。
