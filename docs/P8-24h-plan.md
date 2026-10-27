# P8 — 一昼夜长程任务计划（24 h）

> 日期：2026-10-27。**本文是执行排期的唯一权威**；架构判断见 [`P8-architecture-review.md`](./P8-architecture-review.md)，MoE 方案见 [`P8-moe-fusion-and-balance.md`](./P8-moe-fusion-and-balance.md)，指标规范见 [`P8-benchmark-spec.md`](./P8-benchmark-spec.md)。
> 规则：**每个实验先写门再跑**（假设 → 唯一变量 → 预期 → 判定门 → 实际 → 决策）；一次只跑一个 GPU 任务；失败/负结果照样入报告；单问题 45 min 止损。
> **更新（10-27 19:50，P0–P2 已完成）**：I19 已判定 **launch-bound**（`device_busy` 0.478–0.603、2,401 launches/step，见 `benchmarks/P6-layer-timing.md`）；I8 已有效复测（三档 SLO 0%，瓶颈 TTFT，见 `benchmarks/P6-i8-long-context.md`）；I4 已转正、paged 为默认；投机解码已补离线上界（`benchmarks/P7-spec-decode-addendum.md`）。因此 P8 的 **P1 从"定性 I19"改为"用该判定做 norm/rope 融合与 host-sync 的 A/B"**，§1.1 的“未提交改动”均已提交（`a296069`）。
> **评审修订（10-27 20:10，已批准）**：
> 1. **删除投机解码 Track**：模型草稿路线已两次否决，离线解析上界写死（α 距 1.3× 门 142×，完美草稿 1.45–2.21×）；原 H22–23 的"终裁"没有新信息。见 §10。
> 2. **调度层只补三个硬缺口**：KV 准入/429（I26）、mixed-step 走 decode 路径（I23）、`can_allocate == -1` 的队头阻塞（`scheduler.py:181-184`）。**radix tree、缓存淘汰、小页本轮不做**（radix 列入 §11 条件项）。
> 3. **融合优先、图 4 h 硬停**：fused RMSNorm/RoPE 不依赖图，先做；图只验证 padding-slot 假设。
> 4. **P1/P4 已完成项缩短**：逐层计时与 I8 基线不再占时间；新增 KV 块数一致化（730/728）与图门来源审计两个 P0 项。

---

## 0. 目标与验收

### 0.1 一句话目标

在 24 h 内把四件事做实：**① 拿到至少一项确定性提速（fused RMSNorm/RoPE 或 host-sync）；② 给图 replay 漂移一个可验证的根因判决（4 h 硬停）；③ 修掉 I8 暴露的三个调度硬缺口并用有效重测证明；④ 把 benchmark 升到 v3 并用它重测容量**。MoE 做到"有真实测量 + 有判决"，其中容量收缩优先于放置与融合。

### 0.2 验收（缺一不可）

| # | 验收项 | 判定 |
|---|---|---|
| A | 提速 | 至少一项修复（fused RMSNorm/RoPE、sampler/prefill host-sync、bmm_dense 容量收缩）经交错重复 A/B：token 门通过且目标指标下降 ≥8%（CV ≤5%） |
| B | 图 | 要么 padding-slot 根因修复后 tiny 5/5 + 8B 5/5 + 30B 过门；要么给出"不是这个根因"的证据并停止（4 h 硬停） |
| C | MoE | 真实负载（fold 前计数）的 per-expert/rank 负载 + 容量收缩判决 + `linear` vs `profiled` A/B + route/pack 微基准；每项有判决 |
| D | 长上下文 | 三个调度硬缺口修复后，1k token 重测且 **50% 档 TTFT p50 ≤2 s**；2k 单档有效（或明确"无效 + 根因"）；rank1 全程存活 |
| E | 服务化 | v3 口径的固定并发曲线 + Poisson 四档 + 100% 档 3 次重复 CI；两套 SLO 并列；容量结论 |
| F | 收尾 | 所有证据 JSON/日志有 sha256；ISSUES/PLAN/HANDOFF 更新；本地 commit（**不 push**）；双卡 0 MiB、无残留进程、`/dev/shm` 干净 |

---

## 1. 起点（2026-10-27 20:10，已核实）

### 1.1 代码状态

- 分支 `feat/musa-port`，HEAD **`6cac1eb`**（P0–P2 的代码在 `a296069`，文档与报告在 `d5779b6`/`c552d84`/`6cac1eb`）。
- 工作区干净（仅 `.DS_Store` 未跟踪），**无需再提交基线**；启动前用 `git log -1 && git status --short` 核验即可。
- 关键文件：`nanovllm/config.py`（paged 默认）、`model_runner.py`（`NANOVLLM_TRACE_INIT`）、`api_server.py`（faulthandler、`--max-model-len`、flip 诊断）、`scripts/benchmark_online.py`（`--prompt-tokens`、`slo_success_rate`）、`tests/p6_layer_timing.py`、`scripts/p7_spec_ceiling.py`、`scripts/p2_write_reference.py`。

### 1.2 引擎栈（生产口径）

`attention=triton_paged`（默认）、`moe_backend=grouped_mm`（prefill）、`decode_moe_backend=bmm_dense`、`enforce_eager=True`、`scheduler=stall_free`（在线）/`fcfs`（离线）、TP2/EP2、Qwen3-30B-A3B-Instruct-2507-FP8（加载期反量化 bf16）。

### 1.3 已达成数字（作为回归基线）

| 项 | 数字 | 证据 |
|---|---|---|
| 离线 b32 | 110.6 tok/s（sdpa+bmm）；**133.0（paged+bmm，来自 P6 eager matrix，未单独报告/未固化为基线）** | `benchmarks/P6-nightly.md` §1、`docs/HANDOFF-P6c.md:17` |
| 离线 b64 | 141.48 tok/s（sdpa+bmm；**paged+bmm 未测**） | `benchmarks/P6-i5-i6-i7.md` §1 |
| 在线 v4 | 四档 100% SLO；100% 档 TPOT p95 0.327 s、TTFT p95 1.203 s、goodput 0.776 | `benchmarks/P6-serving-v4.md` |
| 硬饱和（v3 口径） | 0.94 req/s / 60 tok/s；SLO 软边界 0.87–0.96 | `benchmarks/P6-nightly.md` §4 |
| KV | 730/728 blocks（**不一致，见 P0 卫生项**）、12 MiB/block、≈186,880 token/rank、bf16 | `benchmarks/P6-serving-v4.md` §4 |
| 30B 启动 | 热到 `/health` ≈377 s（FP8 反量化 351 s）；冷 11–12 min | `benchmarks/P6-serving-v4.md` §5 |
| I19 逐层 | 0.6B TP1 eager b1：步时 53.7–61.2 ms、device_busy 0.478–0.603、2,401 launches/步、13.5 µs/次 | `benchmarks/P6-layer-timing.md` |
| I8 长上下文 | 384/384、rank1 存活；三档 SLO 0%；TTFT p50 3.84/8.28/10.22 s；空闲单请求 TTFT 0.368 s | `benchmarks/P6-i8-long-context.md` |

### 1.4 服务器

`ssh -p 28036 root@connect.sha1.seetacloud.com`（别名 `s4000b`）。代码 `/root/moe-server-s4000`（伪 `.git`，**只用 scp/rsync 同步，不在远端 git 操作**）。模型：30B `/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8`；0.6B `/root/autodl-tmp/models/Qwen3-0.6B`（若路径不同先 `ls`）；8B `/root/models/Qwen3-8B`。

---

## 2. 资源与约束

### 2.1 30B 加载账本（4 发 + 每次 5 min 收尾）

| # | 时段 | 用途 | 期望产出 |
|---|---|---|---|
| L1 | H6.0–H7.5 | 图 30B A/B（仅当 tiny+8B 5/5） | `benchmarks/data/p8/p8_graph_30b.json` |
| L2 | H8.0–H11.5 | MoE：容量收缩、真实负载、placement、fused_grouped_mm、mixed step | `p8_moe_capacity.json`、`p8_moe_balance.json` |
| L3 | H13.0–H15.5 | 长上下文 1k/2k（硬缺口修复后）+ chunked prefill A/B | `p8_long_context.json` |
| L4 | H16.0–H19.0 | 服务化 v5：并发曲线 + Poisson + 3×repeat | `p8_serving_v5.json` |

- **每发预算 = 加载 + 实验 + 收尾**：热加载 6–8 min、冷 13–15 min、teardown（I12 卡死时按 PID kill + 清 shm）**5 min**。账本按冷加载 15 min 备，余量用于文档。
- 单发加载内可做多个实验（用 `model_runner.call(...)` 切后端/策略），但**一次只跑一个进程**。
- 若某发加载失败：原样记录，最多重试 1 次；仍失败则跳过该 Track，转文档与离线工作。
- **0.6B / 8B 实验不占 30B 账本**（H1.5–4、H4–8 的 tiny/8B 段）。

### 2.2 时间盒与噪声协议

- 单问题 **45 min**：到点写"已排除 / 未排除 / 下一步"，切下一条。
- 图 Track **4 h 硬停**（H4–H8）：到点无论结果都转 MoE。
- MoE placement **2 h 硬停**；融合只到"微基准 + 设计"，不在 24h 内改引擎主路径。
- **噪声协议（0.6B A/B 必用）**：同一进程内交替跑 baseline/fix 各 ≥5 次，取中位数与 CV；**CV >5% 的 A/B 判 INVALID**，不得用单次差值下结论。依据：本轮实测同一配置步时在 **53.7–61.2 ms** 间波动（12%），单次对比的 −8% 可以完全由噪声产生。
- **代码先行铁律（最重要）**：任何需要改引擎代码的实验，**代码、单测、tiny 验证必须在加载 30B 之前完成**。加载窗口只做测量，不做"边改边测"——改一行代码就要重载 6–15 min，24 h 里最多只够 4 次加载。具体：
  - L1 前（H6.5）：图修复已在 tiny/8B 验证；
  - L2 前（H8.0）：容量收缩代码 + tiny 单测已完成；
  - L3 前（H13.0）：三个调度硬缺口 + 单测已完成；
  - L4 前（H16.0）：benchmark v3 的 `--mode concurrency` / `--repeats` 已完成。
  代码没就绪 → **不要加载**，用 buffer 补；加载后才发现要改代码，判该 Track 失败并记录。
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

**并行会话检查（每发必做）**：仓库里可能有另一个会话在同一台机器上跑 GPU。开跑前除 `GPU_BUSY` 外，再核一次 `mthreads-gmi` 双卡 0 MiB 与本会话 lock；若发现陌生 python 进程，**不要杀**，先在 `docs/NIGHTLY-LOG.md` 记录并暂停，等对方结束。收尾时也只清理自己 PID 记录过的进程。

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
| H0–0.5 | P0 | 预检、**KV 块数一致化（I21）**、tiny 回归 | 干净基线 | 两 rank blocks 相同；六配置 token 全等、rc=0 |
| H0.5–1.5 | P0 | 实现 benchmark v3 最小集（queue、token-goodput、双 SLO、timeout、429、p99） | `scripts/benchmark_online.py` v3 | 单测通过；旧 JSON 可回放 |
| H1.5–3.5 | P1 | **fused RMSNorm** 实现 + 0.6B 交错 A/B（RoPE 为可选，H3.5 未绿即顺延） | `p8_fusion.json` | token 全等；b1 −≥8%；CV≤5% |
| H3.5–4.0 | P1 | sampler/prefill host-sync 修复（0.6B 先验；30B 在 L2 复核） | `p8_sync_fix.json` | token 全等；b32 sample −≥50% |
| H4.0–4.5 | P2 | 复现 tiny 漂移 3/3；**图门来源审计**（跑前写死双轨判定） | `p8_graph_repro.json` | 3/3 复现（否则记"不可复现"） |
| H4.5–6.5 | P2 | scratch slot + **slot 0 K/V 回读** + 共享前缀；tiny 5/5 → 8B 5/5 | `p8_graph_padfix.json` | tiny/8B 各 5/5 token 全等；slot0 K/V 逐元素相等 |
| H6.5–7.5 | P2 | **L1**：30B 图 A/B（b1/b8/b32，5 次） | `p8_graph_30b.json` | 硬门 token 全等；性能按 §4.3 双轨 |
| H7.5–8.0 | P2 | 判决与报告 | `benchmarks/P8-graph.md` | 过门 / 明确否决 |
| H8.0–8.5 | P3 | **L2** 启动；**fold 前** expert 计数 | `p8_moe_balance.json` | 每层 128 expert 计数、rank 不平衡 |
| H8.5–10.5 | P3 | **bmm_dense 容量收缩**原型 + A/B | `p8_moe_capacity.json` | 自确定 3/3；L2 软门；MoE −20% |
| H10.5–11.5 | P3 | `linear` vs `profiled` A/B（容量收缩开启） | 同上 | 吞吐 +5% 或 imbalance −10% |
| H11.5–12.0 | P3 | fused route/pack 微基准（只出设计） | `p8_moe_fusion.json` | 微基准 −30%；不改主路径 |
| H12.0–13.0 | P4 | **三个硬缺口**：KV 准入/429、mixed-step decode 路径、HOL 修复 | 代码 + 单测 | 不可准入不阻塞后续；429 计数正确；mixed 走 decode 后端 |
| H13.0–15.0 | P4 | **L3**：1k 重测三档 + chunked on/off | `p8_long_context.json` | **50% 档 TTFT p50 ≤2 s**；有效门见 §4.5 |
| H15.0–15.5 | P4 | 2k 单档（64 并发） | 同上 | 有效或明确 preemption 占比 |
| H15.5–16.0 | P4 | 判决与报告 | `benchmarks/P8-long-context.md` | 改进门过 / 否 |
| H16.0–16.5 | P5 | **L4** 启动；v3 客户端自检 | `p8_serving_v5.json` | health + 冒烟 |
| H16.5–18.0 | P5 | 固定并发曲线 1/4/8/16/32/64 | 同上 | 每点 128 请求，错误分桶 |
| H18.0–19.0 | P5 | Poisson 四档 + 100% 档 3 次重复 | 同上 | 双 SLO 并列 + CI |
| H19.0–20.0 | buffer | 补测 / 修报告 / 未完成项降级 | — | 不排新实验 |
| H20.0–22.0 | P6 | 各报告定稿 + 证据 sha256 | `benchmarks/P8-nightly.md` | 数字可追溯 |
| H22.0–23.0 | P6 | ISSUES/PLAN/HANDOFF 更新 | `docs/HANDOFF-P8.md` | 完成条件写死 |
| H23.0–24.0 | P6 | 本地 commit（不 push）、服务器清零、最终核对 | commit | 验收 F |

> **注**：H20–24 不再包含投机解码终裁（已删除，见 §10）；空出的时间并入 buffer 与文档。

---

## 4. 各 Track 详细指令

### 4.1 P0 — 基线核验 + KV 块数一致化 + 指标 v3（H0–1.5，无 30B）

**P0.1 基线核验（H0–0.25）**

```bash
cd /Users/tmchao7/repos/02_personal/moe-server-s4000
git log -1 && git status --short          # 期望 HEAD 6cac1eb，仅 .DS_Store
ssh s4000b 'cd /root/moe-server-s4000 && md5sum nanovllm/config.py nanovllm/entrypoints/openai/api_server.py scripts/benchmark_online.py'
```

**P0.2 KV 块数一致化（H0.25–0.5，I21）**

现状：rank0 730 blocks、rank1 728（`[kvcache]` 行）。调度器（rank0）按 730 分配，rank1 的块 id 728/729 会越界。修法：

1. `allocate_kv_cache` 后做一次 `all_reduce(min)`（或 `dist.broadcast` rank0→rank1 的较小值），两 rank 用同一个 `num_kvcache_blocks`；`config.num_kvcache_blocks` 在 rank 间保持一致。
2. 门：两 rank 的 `[kvcache] blocks=` 相同；构造一个接近满容量的请求（block id 接近上限）跑 8 token，token 正常、无 MUSA 错误。

tiny 回归：

```bash
ssh s4000b 'cd /root/moe-server-s4000 && PYTHONPATH=. python3 tests/p2_grouped_mm.py --tp 1'
ssh s4000b 'cd /root/moe-server-s4000 && PYTHONPATH=. python3 tests/p6_bmm_dense_tiny.py'
```

门：六配置逐 token 全等、rc=0（graph 段默认不跑）。

**P0.3 指标 v3（H0.5–1.5）**

按 [`P8-benchmark-spec.md`](./P8-benchmark-spec.md) §4/§5 实现，最小集：

1. `summary` 增加 `queue_p50/p95_s`、`goodput_tokens_per_s`、`slo_strict_rate`（`slo_standard` 且 `max ITL ≤1 s`；**不用 ITL p95 当 SLO**，64 token 的 p95 只有一个样本）。
2. 请求记录增加 `error_kind ∈ {none, timeout, http, reject_429, stream, token_count}`；`--request-timeout-s`（默认 120），超时必须调用 `engine.cancel`。
3. `--repeats N`：每档重复 N 次，summary 加 `slo_standard_rate_ci95` / `slo_strict_rate_ci95`（Wilson）与 run-to-run 极差。
4. `--mode concurrency --concurrency 1,4,8,16,32,64`：固定并发（每并发一个 worker，完成即发下一条），与 Poisson 模式并列。
5. `--prompt-tokens` 成为长上下文/服务化实验的默认入口；`prompt_mode=mixed` 支持长度分布（短/中/长 3:4:3）。
6. `prefix_cache_hit_rate` = `kv_prefix_hit_tokens_total` 增量 / prompt tokens 总和；**shared 与 unique 两种负载分开报**。
7. 每 2 s 采样 `mthreads-gmi` 到 `gpu_samples`（仅记录 rank0/1 显存与 util）；warmup 阶段单独标记，**不计入首档**（paged 首次 decode 会 JIT）。
8. p99 只在每点 ≥500 请求时下结论；128 请求的 p99 标 `"p99_reference_only": true`。

门：`python3 -m pytest tests/test_benchmarking_v3.py`（新建，纯函数：Wilson、token goodput、ITL SLO、error 分桶、schema 回放）全绿；用 v4 JSON 回放不抛异常并标 `schema_version: 2`。

### 4.2 P1 — Kernel 融合 + host-sync（H1.5–4，0.6B 为主）

**依据（已测，不再重测）**：0.6B 一步 2,401 次 launch、13.5 µs/次、GPU 空转 40%。按 kernel 类型逐项对到源码：

| 来源 | 每层 launch（约） | 28 层合计 | 代码 |
|---|---:|---:|---|
| RMSNorm ×2（fp32 上采样：cast/pow/mean/add/rsqrt/mul/cast） | ~18 | **~1,000** | `layernorm.py:24-38` |
| RoPE（chunk + 4 乘加 + cat + cast，q/k 各一遍） | ~18 | ~500 | `rotary_embedding.py:12-15` |
| SDPA decode 的 mask/gather | ~6 | 168 | `attention.py:160-192`；**paged 后 decode 已消失** |
| GEMV/GEMM | ~4 | 113 | 真正算力，合计 ~3.4 ms |

**P1.1 fused RMSNorm（必须做，H1.5–3.5）**

- 新增 Triton kernel：bf16 输入/输出，**fp32 内累加**，一次完成 `(x+residual) → rmsnorm → weight`，同时返回新的 residual；替换 `add_rms_forward` 与 `rms_forward` 两条路径。
- 数值门：与现有实现的 logits `max_abs < 2e-3`（L1 口径）+ top-1 翻转率 <0.5%（L2 口径）；两实现各自 3 次确定。
- 性能门（**交错 A/B**）：baseline/fused 在同一进程交替各 5 次，0.6B b1 decode 步时中位数 **−≥8%**，且 CV ≤5%；否则判 INVALID，不得用单次差值。
- 若 H3.5 仍未绿：停，保留测量与代码分支（env flag 默认关），不硬凑。

**P1.2 fused RoPE（可选，H3.5 后顺延）**

- cos/sin 查表 + 单 kernel 旋转，替换 `apply_rotary_emb` 的 chunk/mul/add/cat/cast。门同上。未在 H3.5 前启动则顺延到下一轮，不挤图与 MoE。

**P1.3 host-sync 修复（H3.5–4.0）**

1. **prefill 的 96 次 `.tolist()`**：在 `prepare_prefill`/`set_context` 时把 `cu_seqlens_q/k` 转成 Python list 存进 Context，`_forward_prefill` 直接用；不改数值顺序。
2. **sampler 的 2 次 `bool()`**：`Sampler.forward` 增加 `all_greedy` 参数，由 host 侧 `temperatures` 列表算出；全贪心直接 `argmax`。
- 门：token 全等；b32 sample 时间 **−≥50%**（30B 在 L2 复核）。

证据：`/root/p8_fusion.json`、`/root/p8_sync_fix.json` + `benchmarks/P8-i19.md`。

### 4.3 P2 — 图重放根因（H4–8，4 h 硬停）

**假设（先写进日志）**：replay 时 padding 行 `slot_mapping=0`，每个 bucket 多跑的行把 KV 第 0 个 slot 覆写；block 0 常属于活跃请求或前缀缓存共享块 → 首个 replay 漂移。

**P2.1 复现（H4–4.5）**

```bash
ssh s4000b 'cd /root/moe-server-s4000 && PYTHONPATH=. python3 tests/p6_graph_min.py --tp 1 --generates 3'
```

- 记录 **捕获失败与 replay 漂移分开计数**（I3：捕获阶段有过 `operation not permitted`；两者不是同一个故障）。
- 若 3/3 一致（不复现）：记录"当前变量组合不可复现"，仍可做 P2.2 的修复，但 30B A/B 取消。

**P2.2 修复与验证（H4.5–6.5）**

方案 A（首选）：**预留 scratch slot**。KV 张量多分配 1 个 block（不交给 BlockManager），replay 前把 padding 行的 `slot_mapping` 指向该 block 的第 0 个 slot；同时每次 replay 把 `block_tables[bs:]` 清零，避免 padding 行读到其他请求的块。

方案 B（备选）：给 `store_kvcache` 加 `write_mask`（`slot_mapping=-1` 会写末尾，不可用）；用 `torch.where` 把 pad 行映射到 scratch。

**三个验证门（缺一不可）**：

1. **slot 0 回读**：replay 前把 KV slot 0 的 K/V 拷一份，连续 replay 3 次后逐元素比较（`max_abs == 0`）。只比 token 无法证明 slot 0 没被写。
2. **共享前缀**：两条请求共享 ≥256 token 前缀（命中 block 0）在 graph 下 token 全等；这是 block 0 最可能被复用的场景。
3. **JIT 预热**：paged 是惰性 JIT。`capture_cudagraph` 的 warmup 必须确认走到 `_forward_decode`（decode 上下文），否则 `capture_begin` 会撞上 Triton 编译（历史 "operation not permitted"）。

顺序：tiny 5/5 → 8B 5/5。**8B 不过，禁止 30B**。

**P2.3 图门来源审计（H4.5 前写死，不可跑后改）**

`b1 ≥+15%` 来自 bmm_dense 采用**之前**的 `GRAPH_GATES`；现在 eager 已拿走 decode 的大头，图增量必然变小。跑 30B 之前必须把判定写成双轨：

- **硬门（不变）**：图与 eager 逐 token 全等（b1/b8/b32）；吞吐 CV ≤5%；图显存增量 ≤2 GiB。
- **性能门（双轨）**：b1 **≥+15%** → 转正；**+8%~+15%** → 记录为"正确性门通过、性能待产品决策"，不擅自改默认；**<+8%** → 判"图在本栈不划算"。
- b8 ≥+5%、b32/b64 ≥0.98 仅报告，不作为 30B 转正的唯一否决项（bmm_dense 后 b32 的图收益本就接近 0）。

**P2.4 30B A/B（H6.5–7.5，L1）**

配置：`attention=triton_paged`、`decode_moe_backend=bmm_dense`、`enforce_eager=False`、`max_num_seqs=32`、显式 `num_kvcache_blocks` 留 graph 头寸。

- 全过 → 更新默认（仅 `decode_moe_backend=bmm_dense` 时自动启用），写 `benchmarks/P8-graph.md`。
- 不过 → 默认保持 eager，写"否决 + 证据"；4 h 到点即停，不做第二次。

### 4.4 P3 — MoE：容量收缩 → 放置 → 融合（H8–12，L2）

**P3.1 真实负载统计（H8–8.5）**

用 32 条真实 chat prompt（中英各 16），prefill 512、decode 64，`collect_expert_stats=True`。记录每层 `expert_counts`，计算：

- `expert_load_max/mean`、Gini；
- `rank_load_imbalance = max(rank_slots)/mean - 1`；
- **必须在 fold 之前计数**：在 `topk_ids` 处插桩，统计每个 expert 的**真实**路由数；`local_pairs` 只含 owned slot，看不到 expert 0 的折叠倾斜，**不可用**。
- 对照：把非本 rank slot 计入 expert 0 的实际工作量，与 synthetic（随机 token）的 1.7–5.2% 对比，判断是否被低估。

**P3.2 bmm_dense 容量收缩（H8.5–10.5，最高 ROI）**

- 现状：每个 expert 按 `S = T×K` 做 dense bmm（b32 = 256 行），实际只需 `counts.max()`（估 8–16 行）。设计见 `P8-moe-fusion-and-balance.md` §3（owned-only + 动态 S）。
- 数值门（**不要**用"与现有 bmm_dense 逐 token 全等"——bmm 形状变了，累加顺序会变）：自身确定性 3/3 + 对 sdpa/grouped 参考走 L2 软门（top-1 翻转率 <0.5%、翻转处 margin <0.5）。
- 性能门：b32/b64 的 MoE 段 **−≥20%**，decode 步时 **−≥10%**；交错 ≥5 次、CV ≤5%。
- 不过门 → 记录数字，保留设计，不改默认。

**P3.3 放置 A/B（H10.5–11.5）**

用真实负载生成 `p8_expert_map.json`（`scripts/profile_experts.py` 的 LPT），在**容量收缩开启**的前提下跑 `linear` vs `profiled`：b32 decode ×3、b32 prefill 512 ×3。门：吞吐 **+5%** 或 rank 不平衡 **−10%**；否则关闭 profiled（写"不采用"）。

**P3.4 fused route/pack 微基准（H11.5–12.0）**

- `fused_grouped_mm` vs `grouped_mm`：b32 decode、prefill 512/2048；门：MoE 段 −≥5%，数值走 L1/L2，两后端各 3 次确定。
- route/pack 微基准：测 `argsort/index_select/counts/cumsum` 占比，与 558.8 ms 对齐；**只产出设计**（24h 不实现新 kernel）。时间富余才做原型，门是微基准 −≥30% 且 token 全等。

> mixed step 不再在 MoE Track 里"测代价"——它已被提升为 P4 的**修复项**（I23），门在 §4.5。

报告：`benchmarks/P8-moe.md`，每项一句结论 + 数字 + 证据文件。

### 4.5 P4 — 三个调度硬缺口 + 长上下文重测（H12–16，L3）

**P4.1 三个硬缺口（H12–13，实现 + 单测）**

1. **KV 准入/背压（I26）**：`add_request` 校验 `prompt_len + max_tokens <= max_model_len`，超限返回 400；`schedule()` 容量不足时返回空批次并让请求继续排队，不 assert；`AsyncLLMEngine` 在队列/容量超限时返回 **429** 并计数 `requests_rejected`。
2. **mixed-step 走 decode 路径（I23）**：同一个 step 内把 batch 拆成 prefill 子批与 decode 子批（或给每条 seq 打 `is_prefill` 标签），让 decode 行走 `prepare_decode` + paged + bmm_dense，prefill 行走 dense 路径；注意力/MoE 分别调用。门：mixed 步的 decode 行与纯 decode 步的 token 全等；在线 TPOT 改善（在 P4.3 测）。
3. **队头阻塞（HOL）**：`_schedule_stall_free` 当前在 `can_allocate == -1` 时 `break`（`scheduler.py:181-184`），一个不可准入的长 prompt 会阻塞后面所有 prefill。改为**跳过该候选、继续尝试其他等待请求**（或允许分块预留），并记录 `kv_admission_skips_total`。

单元门：构造"长 prompt 不可准入 + 短 prompt 可准入"的队列，短 prompt 必须在本步被调度；429 路径有计数；mixed 步 decode 行 token 与纯 decode 全等。

**P4.2 有效门 + 改进门（先写后跑）**

有效门（与上一轮相同）：

- `prompt_tokens` p50 ∈ **[960, 1088]**（1k 档）/ **[1920, 2176]**（2k 档）；
- watchdog 全程 rank1 存活；
- ≥125/128 请求收到 `[DONE]`；
- 任一不满足 → 记"无效跑"，保留现场，最多重试 1 次。

**改进门（新增，这才是本轮的目的）**：

- **1k 档 50% 负载：TTFT p50 ≤ 2.0 s**（上一轮 3.84 s）；且 SLO ≥99%。
- TPOT p50 在 50% 档 ≤0.5 s（上一轮 0.418 s 已过，作为回归）。
- 不达 → I26/I23 判"未解决"，把现场数字写进报告，不再重跑同一组合。

**P4.3 运行（H13.0–15.5）**

1. 1k 档：Poisson `--max-qps 0.87`，fractions 0.5/0.75/1.0，128 请求，64 输出；`--prefill-chunk-size 256` vs `2048`（唯一变量）。
2. 2k 档：单档 0.5、**并发降到 64**（186,880 token/rank × 64×2k = 128k，留余量），记录 preemption 次数与 KV 峰值；若 128 并发则只观测 preemption，不评价 SLO。
3. 第二负载（若时间够）：**shared-prefix 长 prompt**（前 512 token 共享、后 512 unique），用于区分"长上下文本身"与"前缀复用缺失"；这一档不参与改进门，只报告 prefix hit rate 与 TTFT。

报告：`benchmarks/P8-long-context.md`，含 TTFT/TPOT/ITL/E2E p50/p95/p99、两套 SLO、goodput、preemptions、KV 峰值、chunked on/off、shared/unique 对比。

### 4.6 P5 — 服务化 v3/v5（H16–19，L4）

**口径约束（先写死）**

- **两套 SLO 并列报告**：请求级 TPOT（与 v3/v4 可比）与 ITL p95（v3 新口径）。容量结论必须注明用的是哪套；只报一套视为口径漂移。
- **p99 仅参考**：128 请求的 p99 实际是第 2 大值，波动极大；要下 p99 结论就把每点请求数提到 ≥500，否则只报 p95。
- **超时必须取消**：客户端 `--request-timeout-s` 到期后调用 `engine.cancel(seq_id)`，否则服务端仍占 KV slot；记录 `timeout` 桶。
- **warmup 排除 JIT**：服务端首次 decode 会触发 paged Triton 编译，4 条 warmup 必须全部完成后再开始计时；报告里注明是否已预热。

**P5.1 固定并发曲线（H16.5–18.0）**

`--mode concurrency --concurrency 1,4,8,16,32,64 --requests 128 --max-tokens 64 --prompt-tokens 128`。每点后查 `/health`；错误按 `{none, timeout, http, 429, stream, token_count}` 分桶。

**P5.2 Poisson 四档 + 重复（H18.0–19.0）**

与 v4 同参（`--max-qps 0.87 --fractions 0.5,0.75,0.9,1.0 --requests 128 --prompt-words 128`），用 v3 客户端；100% 档 **3 次重复**，另两档各 2 次（用于 run-to-run 极差）。

**P5.3 容量判决（H19.0）**

- 容量定义：**在重复中 SLO 达标率均 ≥99% 的最大达成 QPS**，给 Wilson 95% CI 与 run-to-run 极差。
- 回归门：100% 档（0.87）请求级 SLO 100%，TPOT p95 ≤ **0.343 s**（v4 0.327×1.05），TTFT p95 ≤ **1.263 s**（1.203×1.05）。
- 与 v4 同表；若 v3 口径下数字变化 >5%，先解释口径差异（ITL vs TPOT、queue 是否计入）再下结论。

报告：`benchmarks/P8-serving-v5.md`（v4/v5 同表 + 并发曲线 + 错误分桶 + 双 SLO）。

### 4.7 P6 — 收口（H20–24）

1. `benchmarks/P8-nightly.md`：成绩总表 + 每份证据 sha256。
2. **投机解码：已删除，不排期**（见 §10）；不写终裁、不重跑、不占 GPU。
3. `docs/ISSUES.md` 更新：I4/I8 已是关闭状态（无需再改）；新增或更新 I22（graph padding 写 slot 0）、I23（mixed step 后端，本轮修复）、I24（host sync，本轮部分修复）、I25（benchmark v3）、I26（KV admission/backpressure，本轮修复）、I21（per-rank KV 容量，本轮修复）；记录 I19 的融合 A/B 结果。
4. `docs/PLAN.md`：Track 状态、本轮新增的"已删除"与"条件项"。
5. `docs/HANDOFF-P8.md`：30 秒状态、本轮成绩、未解问题、下一轮建议、SOP、红线。
6. 本地 commit（不 push）；服务器清零核对（双卡 0 MiB、无残留、shm 干净）。
7. **buffer**：H19–20 与 H22–24 的空档只用于补测/修报告，不启新实验；任何 Track 超时按 §6 降级。

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
| 1k 改进门未过（TTFT p50 >2 s） | I26/I23 判"未解决"；不重跑同一组合，写现场 |
| 融合 A/B 的 CV >5% | 判 INVALID；增至 10 次交错，仍不过则放弃该 kernel |
| 图只到 +8~15% | 记为"正确性通过、性能待决策"，不改默认，转 MoE |
| 容量收缩的 L2 软门不过 | 保留设计分支（默认关），不改默认 |
| KV 块数修复后两 rank 仍不一致 | 停止 L3（越界风险），先修再跑 |
| 检测到并行会话占用 GPU | 暂停本 Track 并记录；**不要杀陌生进程** |
| 服务化数字与 v4 差异 >5% 且无法解释 | 不宣布容量提升；只报 v3 口径与 CI |
| 时间不足 H20 | 优先保证证据 sha256 + commit + 清零；报告可留"待分析"标记 |

---

## 7. 风险表

| 风险 | 概率 | 影响 | 处理 |
|---|---|---|---|
| 图根因不是 padding slot | 中 | P2 白花 4 h | 4 h 硬停；失败也是结论（排除一个强假设） |
| profile 在 MUSA 上不可用 | 低 | I19 只剩 A/B 墙钟 | 用逐层 sync span 分类；profiler 失败不阻塞 |
| `fused_grouped_mm` 数值门不过 | 中 | 融合结论为"不采用" | 记录数字；不做猜测性调整 |
| 长上下文触发 KV 上限 | 高 | 2k 档可能大量 preempt | 2k 降到 64 并发；只观测 preemption |
| 并发曲线把服务打崩 | 中 | 时间浪费 | 先跑 1/4/8，再上 16/32/64；每点后查 health |
| 30B 加载冷缓存 15 min | 中 | 排期滑 | 账本按冷 15 min + teardown 5 min 备；慢时并行做文档 |
| **0.6B 步时 12% 噪声** | 高 | 单次 A/B 假阳性 | 交错 ≥5 次、中位数、CV≤5%；不达标判 INVALID |
| **图门来源过时（+15% 定于 bmm_dense 前）** | 高 | 正确性通过却被判否决 | §4.3 双轨判定，跑前写死 |
| **KV 块数 730/728 越界** | 中 | 高 block id 读写错/崩溃 | 提到 P0 修复；不一致则停 L3 |
| **容量收缩改变累加顺序** | 高 | 误用"token 全等"门导致假失败 | 用自确定 + L2 软门（§4.4） |
| **并行会话争用 GPU** | 中 | 数字污染/互相崩溃 | 每发前检查；不杀陌生进程 |

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
| A | P1 / P3 | `benchmarks/P8-i19.md`、`benchmarks/P8-moe.md` |
| B | P2 | `benchmarks/P8-graph.md` |
| C | P3 | `benchmarks/P8-moe.md` |
| D | P4 | `benchmarks/P8-long-context.md` |
| E | P5 | `benchmarks/P8-serving-v5.md` |
| F | P6 | commit + 服务器核对 + `P8-nightly.md` |

---

## 10. 已删除 / 不再排期（写清楚，防止下轮重新立项）

| 项 | 结论 | 原因与证据 | 复活条件 |
|---|---|---|---|
| **投机解码（模型草稿：0.6B/1.7B）** | **删除**，不占 24 h | α=0.0056/0.0068，c=0.384/0.386；k=1 实测 0.725×；解析上界：1.3× 门需 α≈0.80，完美草稿也只有 1.45–2.21×。`benchmarks/P7-phase0.md`、`benchmarks/P7-spec-decode-addendum.md` | **同时**满足：I19 固定成本解决（c≤0.1）+ 训练过的头（α≥0.40 @k=4）+ 产品需要降单流延迟；否则不立项 |
| **投机解码"终裁"步骤** | 删除 | 离线再断言已完成（`p7_spec_ceiling.json`），再跑只能复现 | — |
| **I19 逐层计时** | 删除（已完成） | `benchmarks/P6-layer-timing.md`：launch-bound、2,401 launches/步 | — |
| **I8 基线重测** | 删除（已完成） | `benchmarks/P6-i8-long-context.md`：有效跑、三档 SLO 0% | 只在三个硬缺口修复后重测（§4.5） |
| **P0 提交基线/同步服务器** | 删除（已完成） | HEAD `6cac1eb`，工作区干净 | — |
| **"离线 b32 133.0 tok/s（paged+bmm）"的独立报告** | 删除（已有数字，不重跑） | 该数字来自 P6 eager matrix（`HANDOFF-P6c.md:17`、`/root/p6_eager_matrix.json`），但**未单独固化为 paged 基线**；本轮不重测 | 下一轮若要 paged 离线基线，用 `p2_benchmark --attention-backends triton_paged` 正式测一次 |

## 11. 探索方向（技术栈）

| 方向 | 为什么现在做 | 门 / 成本 |
|---|---|---|
| **fused RMSNorm(add)** | 2,401 launches 中约 1,000 来自 fp32 上采样的 RMSNorm；不依赖图 | 0.6B b1 −≥8%、token 门；2 h（§4.2） |
| **fused RoPE** | 约 500 launches；cos/sin 查表 + 单 kernel | 同上；RoPE 顺延到下一轮也可 |
| **bmm_dense 容量收缩** | 每个 expert 按 T×K 做 dense bmm，实际只需 counts.max()；30B 唯一可能改变容量曲线的单点 | 自确定 + L2 软门、MoE −20%；2 h（§4.4） |
| **mixed-step 双路径** | mixed 走 prefill 路径，decode 丢掉 paged/bmm_dense；I8 的 TPOT 恶化直接相关 | mixed decode 行 token 全等 + 在线 TPOT；1 h |
| **paged/chunked prefill** | 长上下文 TTFT 是已证瓶颈，prefill 仍是 SDPA dense gather | 1k 空闲 TTFT、0.435 QPS 下 TTFT p50；下一轮 |
| **分段图**（attention/linear 进图、MoE eager） | 整图 replay 漂移未解时的 fallback | 先 tiny 5/5，再谈 30B |
| **sampler host-sync 去除** | b32 sample 36.6 ms / 13.7%，2 次 `bool()` | b32 sample −≥50% |
| **分段 KV admission + 分块预留** | I8 的队头阻塞与整段预留；本轮只做最小 429，分块预留是下一步 | 改进门见 §4.5 |
| **KV 量化（I20）** | 容量 ×2、带宽减半；会改数值 | L1–L4 + 容量/带宽；**触发：I19 与长上下文收口后**，周级 |
| **batch-invariant decode** | 自有 Triton kernel 可固定归约顺序，解决 I4 的 0.213% 翻转 | 可选，非性能项 |
| **radix tree / 小页（条件项）** | 当前是 256 粒度哈希链、无淘汰；radix 要配小页（16/32）+ 节点 LRU 才有收益，且需改 `paged_attention.py` 的 `page_size==256` 硬断言 | **前置**：先测 shared-prefix 负载的命中率；只有产品是多轮/共享前缀型才立项 |

**明确不做**：字面 MegaMoe（无 FP8 compute/TMA）、DeepEP/all-to-all（PCIe 下贵 8×）、`torch.compile`/inductor（MUSA 无后端，`compat.py` 注释明确）、prompt-lookup（随投机解码一并删除；仅当出现"长逐字复用"负载再单独立项，且不占 24 h）。
