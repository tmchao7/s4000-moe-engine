# P8 — 一昼夜长程任务计划（24 h）

> 日期：2026-10-27。**本文是执行排期的唯一权威**；架构判断见 [`P8-architecture-review.md`](./P8-architecture-review.md)，MoE 方案见 [`P8-moe-fusion-and-balance.md`](./P8-moe-fusion-and-balance.md)，指标规范见 [`P8-benchmark-spec.md`](./P8-benchmark-spec.md)。
> 规则：**每个实验先写门再跑**（假设 → 唯一变量 → 预期 → 判定门 → 实际 → 决策）；一次只跑一个 GPU 任务；失败/负结果照样入报告；单问题 45 min 止损。
> **更新（10-27 19:50，P0–P2 已完成）**：I19 已判定 **launch-bound**（`device_busy` 0.478–0.603、2,401 launches/step，见 `benchmarks/P6-layer-timing.md`）；I8 已有效复测（三档 SLO 0%，瓶颈 TTFT，见 `benchmarks/P6-i8-long-context.md`）；I4 已转正、paged 为默认；投机解码已补离线上界（`benchmarks/P7-spec-decode-addendum.md`）。因此 P8 的 **P1 从"定性 I19"改为"用该判定做 norm/rope 融合与 host-sync 的 A/B"**，§1.1 的“未提交改动”均已提交（`a296069`）。

---

## 0. 目标与验收

### 0.1 一句话目标

在 24 h 内把三件事做实：**① 定性 I19 并拿到至少一项确定性提速；② 给图 replay 漂移一个可验证的根因判决；③ 把 benchmark 升到 v3 并用它重测容量**。MoE 融合/负载均衡做到"有真实测量 + 有判决"，长上下文做到"有有效数据"。

### 0.2 验收（缺一不可）

| # | 验收项 | 判定 |
|---|---|---|
| A | I19 | 给出 `device_busy` 与 `launch_bound / kernel_bound / mixed` 判定；至少一项 host-sync 修复经 A/B 验证（token 全等 + 时间下降） |
| B | 图 | 要么 padding-slot 根因修复后 tiny 5/5 + 8B 5/5 + 30B 过冻结门；要么给出"不是这个根因"的证据并停止（4 h 硬停） |
| C | MoE | 真实负载的 per-expert/rank 负载 + `linear` vs `profiled` A/B + `fused_grouped_mm`/route/pack 微基准；每项有判决 |
| D | 长上下文 | 1k 与 2k token 的 unique prompt 各一档有效跑（或明确"无效 + 根因"）；rank1 全程存活 |
| E | 服务化 | v3 口径的固定并发曲线 + Poisson 四档 + 100% 档 3 次重复 CI；容量结论 |
| F | 收尾 | 所有证据 JSON/日志有 sha256；ISSUES/PLAN/HANDOFF 更新；本地 commit（**不 push**）；双卡 0 MiB、无残留进程、`/dev/shm` 干净 |

---

## 1. 起点（2026-10-27 19:xx，已核实）

### 1.1 代码状态

- 分支 `feat/musa-port`，HEAD `fa0b113`（`42f848a` 的文档后继）。
- **未提交改动**（P0 必须先提交成基线）：
  - `nanovllm/config.py`：`attention_backend` 默认 `sdpa → triton_paged`
  - `nanovllm/engine/model_runner.py`：`NANOVLLM_TRACE_INIT=1` 分段打点
  - `nanovllm/entrypoints/openai/api_server.py`：faulthandler、`--max-model-len`、`NANOVLLM_ATTENTION_FLIP_AFTER_INIT`
  - `scripts/benchmark_online.py`：`--prompt-tokens` 校准、`slo_success_rate`
  - 未跟踪：`benchmarks/P6-serving-v4.md`、`benchmarks/P7-spec-decode-addendum.md`、`scripts/p7_spec_ceiling.py`、`scripts/p2_write_reference.py`、`tests/p6_layer_timing.py`
- `.DS_Store` 不进 commit。

### 1.2 引擎栈（生产口径）

`attention=triton_paged`（默认）、`moe_backend=grouped_mm`（prefill）、`decode_moe_backend=bmm_dense`、`enforce_eager=True`、`scheduler=stall_free`（在线）/`fcfs`（离线）、TP2/EP2、Qwen3-30B-A3B-Instruct-2507-FP8（加载期反量化 bf16）。

### 1.3 已达成数字（作为回归基线）

| 项 | 数字 | 证据 |
|---|---|---|
| 离线 b32 | 110.6 tok/s（sdpa+bmm）；133.0（paged+bmm，未单独报告） | `benchmarks/P6-nightly.md` §1 |
| 离线 b64 | 141.48 tok/s（sdpa+bmm；**paged+bmm 未测**） | `benchmarks/P6-i5-i6-i7.md` §1 |
| 在线 v4 | 四档 100% SLO；100% 档 TPOT p95 0.327 s、TTFT p95 1.203 s、goodput 0.776 | `benchmarks/P6-serving-v4.md` |
| 硬饱和（v3 口径） | 0.94 req/s / 60 tok/s；SLO 软边界 0.87–0.96 | `benchmarks/P6-nightly.md` §4 |
| KV | 730/728 blocks、12 MiB/block、≈186,880 token/rank、bf16 | `benchmarks/P6-serving-v4.md` §4 |
| 30B 启动 | 热缓存到 `/health` ≈ 6 min 17 s；其中 FP8 反量化 351 s | `benchmarks/P6-serving-v4.md` §5 |

### 1.4 服务器

`ssh -p 28036 root@connect.sha1.seetacloud.com`（别名 `s4000b`）。代码 `/root/moe-server-s4000`（伪 `.git`，**只用 scp/rsync 同步，不在远端 git 操作**）。模型：30B `/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8`；0.6B `/root/autodl-tmp/models/Qwen3-0.6B`（若路径不同先 `ls`）；8B `/root/models/Qwen3-8B`。

---

## 2. 资源与约束

### 2.1 30B 加载账本（4 发）

| # | 时段 | 用途 | 期望产出 |
|---|---|---|---|
| L1 | H6.0–H7.5 | 图 30B A/B（仅当 tiny+8B 5/5） | `benchmarks/data/p8/p8_graph_30b.json` |
| L2 | H8.0–H11.5 | MoE：真实负载、placement、fused_grouped_mm、mixed step | `p8_moe_balance.json`、`p8_moe_fusion.json` |
| L3 | H13.0–H15.5 | 长上下文 1k/2k + chunked prefill A/B | `p8_long_context.json` |
| L4 | H16.0–H19.5 | 服务化 v5：并发曲线 + Poisson + 3×repeat | `p8_serving_v5.json` |

- 热页缓存加载 6–8 min；冷 13–15 min。每次启动前按 §2.3 清零。
- 单发加载内可做多个实验（用 `model_runner.call(...)` 切后端/策略），但**一次只跑一个进程**。
- 若某发加载失败：原样记录，最多重试 1 次；仍失败则跳过该 Track，转文档与离线工作。

### 2.2 时间盒

- 单问题 **45 min**：到点写"已排除 / 未排除 / 下一步"，切下一条。
- 图 Track **4 h 硬停**（H4–H8）：到点无论结果都转 MoE。
- MoE placement **2 h 硬停**；融合只到"微基准 + 设计"，不在 24h 内改引擎主路径。
- 每 4 h 做一次状态快照（写 `docs/NIGHTLY-LOG.md` 追加），防止上下文丢失。

### 2.3 每发加载前的清零 SOP

```bash
# 1) 查（锚定，绝不用未锚定的 pgrep -f）
pgrep -fa "^python3|^[s]pawn_main|^/bin/bash .*run_p8" || true
# 2) 按 PID kill -9（不要按名字 kill）
# 3) 清 shm
rm -f /dev/shm/nanovllm*
# 4) 双卡 0 MiB
mthreads-gmi | grep -E "^[01] "
```

长任务一律：`setsid nohup ... > /root/p8_xxx.log 2>&1 &` + lock 文件 + 开跑前 `GPU_BUSY` 检查 + 短连接轮询。**不要留自动重启的 waiter**。

### 2.4 红线

- 不 `git push`；不 reset 工作区；不改 `site-packages`；不改 `scripts/p5_decide.py` 任何阈值。
- 不用 `vllm_fused` 当默认；不用 `mp22`；不全局设 `moe_backend=bmm_dense`（prefill OOM）。
- 不用未锚定的 `pkill -f`/`pgrep -f`；不在远端 `git` 提交。
- 不拿 tiny 的 token 一致性当图门；8B 修复不直接推 30B。
- 图 Track 未过 8B 5/5 前，禁止 30B graph A/B。

---

## 3. 24 h 总表

| 时段 | Track | 动作 | 产出 | 门 |
|---|---|---|---|---|
| H0–0.5 | P0 | 预检、提交基线、同步服务器、tiny 回归 | 干净基线 | 六配置 token 全等、rc=0 |
| H0.5–1.5 | P0 | 实现 benchmark v3（queue、token-goodput、p99、timeout、CI、`--concurrency`、prefix hit、GPU 采样） | `scripts/benchmark_online.py` v3 | 单测通过；旧 JSON 可回放 |
| H1.5–2.5 | P1 | 0.6B 逐层计时（A/B/C 三测量） | `p8_layer_timing_06.json` | 给出 `device_busy` 与分类 |
| H2.5–4.0 | P1 | host-sync 修复（prefill `tolist` 上提、sampler 去 `bool()`）+ 0.6B A/B | `p8_sync_fix.json` | token 全等；0.6B b1 步时 −≥8% |
| H4.0–4.5 | P2 | 复现 tiny 漂移 3/3；写冻结门 | `p8_graph_repro.json` | 3/3 复现（否则记为"不可复现"） |
| H4.5–6.0 | P2 | padding 行改用 scratch slot + 清残留 block_tables；tiny 5/5 → 8B 5/5 | `p8_graph_padfix.json` | tiny/8B 各 5/5 token 全等 |
| H6.0–7.5 | P2 | **L1**：30B 图 A/B（b1/b8/b32，5 次） | `p8_graph_30b.json` | 冻结图门（见 §4.2） |
| H7.5–8.0 | P2 | 判决与报告 | `benchmarks/P8-graph.md` | 过门 / 明确否决 |
| H8.0–8.5 | P3 | **L2** 启动；真实负载 expert 统计 | `p8_moe_balance.json` | 每层 128 expert 计数、rank 不平衡 |
| H8.5–10.0 | P3 | `linear` vs `profiled` A/B（b32 decode / b32 prefill） | 同上 | 吞吐 ≥+5% 或不平衡 −≥10% |
| H10.0–11.0 | P3 | `fused_grouped_mm` A/B + route/pack 微基准 | `p8_moe_fusion.json` | 数值门 + MoE 段 −≥5%（微基准 −≥30%） |
| H11.0–11.5 | P3 | mixed step 后端代价测量 | 同上 | 给出在线 TPOT 影响上界 |
| H11.5–12.0 | P3 | 判决与报告 | `benchmarks/P8-moe.md` | 每项有结论 |
| H12.0–13.0 | P4 | backpressure/429 + 请求 timeout + rank watchdog + 长上下文 runner | 脚本 | 容量不足不 assert；watchdog 可用 |
| H13.0–13.5 | P4 | **L3** 启动（`--prompt-tokens 1024 --max-model-len 4096`） | `p8_long_context.json` | 有效门（见 §4.4） |
| H13.5–15.0 | P4 | 1k token 三档 + chunked on/off | 同上 | 同一变量、同参数 |
| H15.0–15.5 | P4 | 2k token 单档（若 KV 足够） | 同上 | 记录 preemption 与 SLO |
| H15.5–16.0 | P4 | 判决与报告 | `benchmarks/P8-long-context.md` | 有效数据或无效根因 |
| H16.0–16.5 | P5 | **L4** 启动；v3 客户端自检 | `p8_serving_v5.json` | health + 1 条冒烟 |
| H16.5–18.5 | P5 | 固定并发曲线 1/4/8/16/32/64 | 同上 | 每点 128 请求，错误分桶 |
| H18.5–19.5 | P5 | Poisson 四档 + 100% 档 3 次重复 | 同上 | CI 可算 |
| H19.5–20.0 | P5 | 容量判决 | `benchmarks/P8-serving-v5.md` | 见 §4.5 |
| H20.0–22.0 | P6 | 各报告定稿 + 证据 sha256 | `benchmarks/P8-nightly.md` | 数字可追溯 |
| H22.0–23.0 | P6 | 投机解码终裁 + ISSUES/PLAN/HANDOFF 更新 | `docs/HANDOFF-P8.md` | 完成条件写死 |
| H23.0–24.0 | P6 | 本地 commit（不 push）、服务器清零、最终核对 | commit | 验收 F |

---

## 4. 各 Track 详细指令

### 4.1 P0 — 基线 + 指标 v3（H0–1.5，无 30B）

**P0.1 提交基线（H0–0.5）**

```bash
cd /Users/tmchao7/repos/02_personal/moe-server-s4000
git add nanovllm/config.py nanovllm/engine/model_runner.py \
        nanovllm/entrypoints/openai/api_server.py scripts/benchmark_online.py \
        benchmarks/P6-serving-v4.md benchmarks/P7-spec-decode-addendum.md \
        scripts/p7_spec_ceiling.py scripts/p2_write_reference.py tests/p6_layer_timing.py
git commit -m "feat(p8): baseline — paged default (L5 pass), init tracing, prompt-token calibration, spec ceiling addendum"
```

同步服务器：

```bash
tar czf - <改动文件> | ssh s4000b 'tar xzf - -C /root/moe-server-s4000'
ssh s4000b 'cd /root/moe-server-s4000 && md5sum <改动文件>'
```

tiny 回归：

```bash
ssh s4000b 'cd /root/moe-server-s4000 && PYTHONPATH=. python3 tests/p2_grouped_mm.py --tp 1'
ssh s4000b 'cd /root/moe-server-s4000 && PYTHONPATH=. python3 tests/p6_bmm_dense_tiny.py'
```

门：六配置逐 token 全等、rc=0（graph 段默认不跑）。

**P0.2 指标 v3（H0.5–1.5）**

按 [`P8-benchmark-spec.md`](./P8-benchmark-spec.md) §4/§5 实现，最小集：

1. `summary` 增加 `queue_p50/p95_s`、`goodput_tokens_per_s`、`slo_itl_success_rate`（SLO 用 ITL p95 判定）。
2. 请求记录增加 `error_kind ∈ {none, timeout, http, stream, token_count}`；`--request-timeout-s`（默认 120）。
3. `--repeats N`：每档重复 N 次，summary 加 `slo_success_rate_ci95`（Wilson 区间）与 run-to-run 极差。
4. `--mode concurrency --concurrency 1,4,8,16,32,64`：固定并发（每并发一个 worker，完成即发下一条），与 Poisson 模式并列。
5. `--prompt-tokens` 成为长上下文/服务化实验的默认入口；`prompt_mode=mixed` 支持长度分布（短/中/长 3:4:3）。
6. `prefix_cache_hit_rate` = `kv_prefix_hit_tokens_total` 增量 / prompt tokens 总和。
7. 每 2 s 采样 `mthreads-gmi` 到 `gpu_samples`（仅记录 rank0/1 显存与 util）。

门：`python3 -m pytest tests/test_benchmarking_v3.py`（新建，纯函数：Wilson、token goodput、ITL SLO、error 分桶、schema 回放）全绿；用 v4 JSON 回放不抛异常并标 `schema_version: 2`。

### 4.2 P1 — I19 定性 + host-sync（H1.5–4）

**P1.1 逐层计时（0.6B，TP1）**

```bash
ssh s4000b 'cd /root/moe-server-s4000 && PYTHONPATH=. python3 tests/p6_layer_timing.py \
  --model /root/autodl-tmp/models/Qwen3-0.6B --tp 1 \
  --output /root/p8_layer_timing_06.json'
```

判定（脚本内已冻结）：`device_busy = kernel_total_device / baseline_step_wall`；`<0.70 launch_bound`、`≥0.90 kernel_bound`、中间 `mixed`。同时记录 `host_median_per_layer_s`、`device_span_median_per_layer_s`、`kernel_count`、`top_kernels`。

**P1.2 两项修复（先实现、再 A/B）**

1. **prefill 的 96 次 `.tolist()`**：在 `prepare_prefill`/`set_context` 时把 `cu_seqlens_q/k` 转成 Python list 存进 Context，`_forward_prefill` 直接用；不改数值顺序。
2. **sampler 的 2 次 `bool()`**：`Sampler.forward` 增加 `all_greedy: bool | None` 参数；`ModelRunner` 在 host 由 `temperatures` 列表算出 `all_greedy` 并传入；全贪心时直接 `argmax`，不构造 `greedy` 张量。
3. （可选，若时间够）`RMSNorm`/final norm 的 `add` 与下一个 op 合并——只在 profiler 点名后才做。

门（先写后跑）：

- token 全等：0.6B 与 30B 上，修复前后同一 prompt 贪心 128 token 完全一致。
- 0.6B b1 decode 步时中位数 **−≥8%**；prefill 1k 的 step wall **−≥10%**。
- b32 sample 时间 **−≥50%**（30B 在 L2 验证）。
- 不过门：保留修复但默认关闭（env flag），记录数字。

证据：`/root/p8_sync_fix.json` + `benchmarks/P8-i19.md`。

### 4.3 P2 — 图重放根因（H4–8，4 h 硬停）

**假设（先写进日志）**：replay 时 padding 行 `slot_mapping=0`，每个 bucket 多跑的行把 KV 第 0 个 slot 覆写；block 0 常属于活跃请求 → 首个 replay 漂移。

**P2.1 复现（H4–4.5）**

```bash
ssh s4000b 'cd /root/moe-server-s4000 && PYTHONPATH=. python3 tests/p6_graph_min.py --tp 1 --generates 3'
# 期望：3 次 token 不一致（历史 tiny 稳定漂移）
```

若 3/3 一致（不复现）：记录"当前变量组合不可复现"，仍可做 P2.2 的修复，但 30B A/B 取消。

**P2.2 修复（H4.5–6）**

方案 A（首选）：**预留 scratch slot**。KV 分配时多留 1 个 block（不交给 BlockManager），`run_model` replay 前把 padding 行的 `slot_mapping` 指向该 block 的第 0 个 slot；同时每次 replay 把 `block_tables[bs:]` 清零，避免残留。

方案 B（备选）：给 `store_kvcache` 加 `write_mask`（`slot_mapping` 用 `-1` 不行，会写末尾）；用 `torch.where` 把 pad 行映射到 scratch。

验证顺序：tiny 5/5 → 8B 5/5。**8B 不过，禁止 30B**。

**P2.3 30B A/B（H6–7.5，L1）**

配置：`attention=triton_paged`、`decode_moe_backend=bmm_dense`、`enforce_eager=False`、`max_num_seqs=32`、显式 `num_kvcache_blocks` 留 graph 头寸。

冻结门（不可改，来自 `docs/PLAN.md` §3）：

- 图与 eager **逐 token 全等**（同一批 prompt，贪心，b1/b8/b32）；
- b1 吞吐 **≥+15%**；b8 **≥+5%**；b32/b64 **≥0.98**；
- 吞吐 CV **≤5%**；图显存增量 **≤2 GiB**。

**判决**：

- 全过 → 更新 `config`/`api_server` 默认 `enforce_eager=False`（仅在 `decode_moe_backend=bmm_dense` 时自动启用），写 `benchmarks/P8-graph.md`。
- 不过 → 默认保持 eager，写"否决 + 证据"；4 h 到点即停，不做第二次。

### 4.4 P3 — MoE 真实负载 + 融合（H8–12，L2）

**P3.1 真实负载统计（H8–8.5）**

用 32 条真实 chat prompt（中英各 16），prefill 512、decode 64，`collect_expert_stats=True`。记录每层 `expert_counts`，计算：

- `expert_load_max/mean`、Gini；
- `rank_load_imbalance = max(rank_slots)/mean - 1`；
- **折叠修正**：把非本 rank slot 计入 expert 0 的实际工作量（`owned=False` 的 slot 数）。

对比 synthetic（随机 token）的 1.7–5.2%，判断是否被低估。

**P3.2 放置 A/B（H8.5–10）**

用真实负载生成 `p8_expert_map.json`（`scripts/profile_experts.py` 的 LPT），跑：

- `linear` vs `profiled`：b32 decode（128 token）×3、b32 prefill（512）×3；
- 指标：吞吐中位数、CV、rank 不平衡。

门：**吞吐 ≥+5%** 或 **rank 不平衡 −≥10%**；否则关闭 profiled（写"不采用"）。

**P3.3 融合（H10–11）**

- `fused_grouped_mm` vs `grouped_mm`：b32 decode、prefill 512/2048；门：MoE 段 **−≥5%**，数值走 L1/L2（`max_abs ≤2e-3`、翻转率 ≤0.5%），两后端各 3 次确定性。
- route/pack 微基准：用 profiler 或单独计时，测 `argsort/index_select/counts/cumsum` 的占比；与 558.8 ms 的 P5 数字对齐。门：**只产出设计**（24h 不实现新 kernel）；若时间富余，做一版 fused route 原型，门是微基准 −≥30% 且 token 全等。

**P3.4 mixed step（H11–11.5）**

同一 batch 下比较 `mode=decode` 与 `mode=mixed`（1 条 prefill + 31 条 decode）的 forward 时间与 MoE 后端；估算在线 mixed 步占比下的 TPOT 损失上界。

报告：`benchmarks/P8-moe.md`，每项一句结论 + 数字 + 证据文件。

### 4.5 P4 — 长上下文 + 调度硬化（H12–16，L3）

**P4.1 硬化（H12–13，无 GPU）**

1. `add_request` 校验 `prompt_len + max_tokens <= max_model_len`；超限直接 400，不进入调度。
2. KV 不足时：`schedule()` 返回空批次不 assert；`AsyncLLMEngine` 对排队超限返回 429 + `requests_rejected` 计数。
3. 客户端 `--request-timeout-s`（v3 已有）；runner 侧 watchdog：每 5 s 记录 `mthreads-gmi` 与 `[s]pawn_main` PID 到 `/root/p8_watch.log`。
4. 长上下文 runner：`--prompt-mode unique --prompt-tokens 1024 --max-model-len 4096`。

**P4.2 有效门（先写后跑）**

- `prompt_tokens` p50 ∈ **[960, 1088]**（1k 档）/ **[1920, 2176]**（2k 档）；
- watchdog 全程 rank1 存活（不出现显存归零/PID 消失）；
- ≥125/128 请求收到 `[DONE]`；
- 任一不满足 → 记"无效跑"，保留现场，最多重试 1 次。

**P4.3 运行（H13.5–15.5）**

1k 档：Poisson，`--max-qps 0.87`，fractions 0.5/0.75/1.0，128 请求，64 输出；`--prefill-chunk-size 256` vs `2048` 两次（唯一变量）。
2k 档：单档 0.5，记录 preemption 与 KV 使用。

报告：`benchmarks/P8-long-context.md`，含 TTFT/TPOT/ITL/E2E p50/p95/p99、SLO、goodput、preemptions、KV 峰值、chunked on/off 对比。

### 4.6 P5 — 服务化 v3/v5（H16–20，L4）

**P5.1 固定并发曲线（H16.5–18.5）**

`--mode concurrency --concurrency 1,4,8,16,32,64 --requests 128 --max-tokens 64 --prompt-tokens 128`。

**P5.2 Poisson 四档 + 重复（H18.5–19.5）**

与 v4 同参（`--max-qps 0.87 --fractions 0.5,0.75,0.9,1.0 --requests 128 --prompt-words 128`），但用 v3 客户端；100% 档 **3 次重复**。

**P5.3 容量判决（H19.5–20）**

- 容量定义（新，写进报告）：**在 3 次重复中 SLO 达标率均 ≥99% 的最大达成 QPS**；同时给出 Wilson 95% CI。
- 回归门：0.87 档 SLO 100%，TPOT p95 ≤ **0.343 s**（v4 0.327×1.05），TTFT p95 ≤ **1.263 s**（1.203×1.05）。
- 与 v4 同表；若 v3 口径下数字变化 >5%，先解释口径差异再下结论。

报告：`benchmarks/P8-serving-v5.md`（v4/v5 同表 + 并发曲线图 + 错误分桶）。

### 4.7 P6 — 收口（H20–24）

1. `benchmarks/P8-nightly.md`：成绩总表 + 每份证据 sha256。
2. 投机解码终裁：读 `benchmarks/P7-spec-decode-addendum.md`；写"维持 NO-GO + 三条复活条件（I19 解决、α≥0.40 实测、EAGLE3/prompt-lookup 权重就绪）"。
3. `docs/ISSUES.md` 更新：I4 → ✅（L5 通过、默认已切）；新增 I21（per-rank KV 容量不一致）、I22（graph padding 写 slot 0）、I23（mixed step 后端）、I24（host sync）、I25（benchmark v3）、I26（KV admission/backpressure）。
4. `docs/PLAN.md`：Track 状态与下一轮 P0/P1/P2。
5. `docs/HANDOFF-P8.md`：30 秒状态、本轮成绩、未解问题、下一轮建议、SOP、红线。
6. 本地 commit（不 push）；服务器清零核对。

---

## 5. 证据与报告清单

| 文件 | 内容 |
|---|---|
| `benchmarks/P8-graph.md` | 图：复现、根因验证、30B A/B、判决 |
| `benchmarks/P8-moe.md` | 真实负载、placement、fusion、mixed step |
| `benchmarks/P8-long-context.md` | 1k/2k、chunked A/B、SLO、preemption |
| `benchmarks/P8-serving-v5.md` | 并发曲线、Poisson、CI、容量 |
| `benchmarks/P8-i19.md` | 逐层计时、分类、sync 修复 A/B |
| `benchmarks/P8-nightly.md` | 总表 + sha256 |
| `benchmarks/data/p8/*.json/.jsonl` | 原始证据（不进 git） |
| `docs/HANDOFF-P8.md` | 交接 |

每份报告必须包含：**唯一变量、冻结门（先写后跑）、实际值、判定、失败/负结果的原始数字**。

---

## 6. 降级与止损

| 触发 | 降级动作 |
|---|---|
| 图 4 h 到点未过 | 停；保持 eager；写否决报告；转 MoE |
| 8B 5/5 不过 | 不做 30B 图 A/B；把 L1 让给 MoE |
| L1/L2/L3/L4 任一加载失败 2 次 | 跳过该 Track；用 0.6B/8B 或离线工作补 |
| 长上下文仍无效（prompt 长度或 rank1） | 原样记录；用 512-token 档补一条有效数据 |
| 服务化数字与 v4 差异 >5% 且无法解释 | 不宣布容量提升；只报 v3 口径与 CI |
| 时间不足 H20 | 优先保证证据 sha256 + commit + 清零；报告可留"待分析"标记 |

---

## 7. 风险表

| 风险 | 概率 | 影响 | 处理 |
|---|---|---|---|
| 图根因不是 padding slot | 中 | P2 白花 4 h | 4 h 硬停；失败也是结论（排除一个强假设） |
| profile 在 MUSA 上不可用 | 低 | I19 只剩 A/B 墙钟 | 用逐层 sync span 分类；profiler 失败不阻塞 |
| `fused_grouped_mm` 数值门不过 | 中 | 融合结论为"不采用" | 记录数字；不做猜测性调整 |
| 长上下文触发 KV 上限 | 高 | 2k 档可能大量 preempt | 先跑 1k；2k 只做单档观测 |
| 并发曲线把服务打崩 | 中 | 时间浪费 | 先跑 1/4/8，再上 16/32/64；每点后查 health |
| 30B 加载冷缓存 15 min | 中 | 排期滑 | 账本已按 8 min 热估算；冷启动时并行做文档 |

---

## 8. 日志格式（每个实验）

```markdown
### H<小时> <Track> <实验名>
- 假设：
- 唯一变量：
- 预期（数值）：
- 冻结门（跑前写）：
- 实际：
- 判定：PASS / FAIL / INVALID
- 证据：/root/... （sha256 ...）
- 决策 / 下一步：
```

---

## 9. 验收映射

| 验收 | 对应 Track | 判定文件 |
|---|---|---|
| A | P1 | `benchmarks/P8-i19.md` |
| B | P2 | `benchmarks/P8-graph.md` |
| C | P3 | `benchmarks/P8-moe.md` |
| D | P4 | `benchmarks/P8-long-context.md` |
| E | P5 | `benchmarks/P8-serving-v5.md` |
| F | P6 | commit + 服务器核对 + `P8-nightly.md` |
