# P6 夜间迭代报告（2026-10-26 夜 → 10-27 凌晨）

> 执行者：ZCode 夜间自主迭代（接手 HANDOFF-P6b）。时间跨度 23:50–（收尾中），4 次 30B 加载。
> 逐实验的时间线、预冻结门与判定过程见 `docs/NIGHTLY-LOG.md`（协议：先写门再跑）。
> 本页是明早验收的总表；全部结论有服务器证据 JSON/日志背书（sha256 见 §7）。

## 0. 30 秒结论

| 项 | 门 | 结果 | 判定 |
|---|---|---|---|
| **P0 decode=bmm_dense 30B 矩阵** | b16/b32 吞吐比 ≥1.08、CV ≤5%、不慢于 baseline | b16 比 **1.44–1.53**、b32 比 **1.29–1.37**，CV ≤3.3%，无变慢点 | ✅ **采用** |
| **P1 b32 稳定性（shm 修复验证）** | determinism_b32 全稳、b32 ×10 确定 | 4 案例 + 3 栈 ×10 **全部 0 分歧** | ✅ **b32 稳定性已修复** |
| **P1 b64 崩溃** | 3 次不复现 | 3/3 完成，无 MUSA error | ✅ **未复现** |
| **P1 triton_paged 转正** | 冻结门（decide --mode attention） | 性能门大胜（TPOT ratio 0.59）但一致率 0.875<0.99 | ❌ keep_sdpa（分歧是可复现数值差异，非噪声） |
| **P2 在线容量 v3** | 100% SLO 档从 50% 推到 ≥75% | **四档（含 100%）全部 100% SLO**；100% 档 goodput **15.4×** | ✅ **超额达成** |
| **P3 图 replay** | 8B 5/5 绿 → 才准 30B A/B | 8B：**找到候选修复 ①+②，5/5 绿 + 多重放通**；30B：第一次 replay 仍打死设备 | ⚠️ 8B 修复、**30B 未修**，图方向按时间盒关闭 |

**一句话**：decode MoE 换 `bmm_dense` + shm 竞态修复，把 30B 服务从「50% 负载就破 SLO」变成「v2 口径全负载档 100% SLO」；离线 b32 吞吐 69.4 → **133 tok/s（triton_paged+bmm）** 或 **110.6 tok/s（sdpa+bmm，已采用栈）**。图捕获仍可（+0.52 GiB），但 30B replay 挂死未解（8B 已解）。

## 1. P0：30B eager 矩阵（第 1 发加载，`/root/p6_eager_matrix.json`）

同一次加载内 A/B，prefill=grouped_mm，decode ∈ {grouped_mm, bmm_dense}，attention ∈ {sdpa, triton_paged}；prompt 128 / output 32 / repeats 3 / b1..b32（与 P5-final-matrix 同口径，b64 因已知崩溃点排除）。

| batch | sdpa: grouped→bmm (ratio) | triton_paged: grouped→bmm (ratio) | TPOT p95 b32 (sdpa) |
|---:|---:|---:|---:|
| 1 | 3.96→5.82 (1.470) | 4.42→6.60 (1.495) | — |
| 8 | 28.69→41.90 (1.461) | 30.81→45.85 (1.488) | — |
| 16 | 50.11→72.30 (**1.443**) | 54.38→83.20 (**1.530**) | — |
| 32 | 85.79→110.59 (**1.289**) | 97.21→133.00 (**1.368**) | 0.335→0.249 s |

- 验证门：baseline（loop/grouped_mm）与冻结参考逐 token 全等；**bmm_dense 8-token 输出也与冻结参考全等**（软门意外全过）。
- 跨后端 token 一致率（请求级）：b1–b8 100%，b16 81–88%，b32 75–94%——bf16 累加顺序近平局翻转，符合预期，不设门。
- vs P5-final-matrix 默认栈（sdpa+grouped b32=69.39、TPOT p95 0.4244 s）：**已采用栈 110.6 tok/s（+59%）/0.249 s（-41%）；全组合最优 133.0 tok/s（+92%）/0.199 s（-53%）**（triton_paged 未转正，见 §2）。
- 整步更快还有上轮无同步化（route/pack、store_kvcache）的功劳：sdpa+grouped b32 也从 69.4 提到 85.8。

## 2. P1：D2 稳定性 + attention 决策（第 2 发加载，`/root/p6_d2_nightly.json`、`/root/p6_attention_ab.json`）

一次加载跑完全部（合并脚本 `tests/p6_d2_nightly.py`）：

1. **b32 不可复现已修复**：determinism_b32 四案例（A grouped_b32 / B loop_b32 / C zero-KV / D b16）全稳；b32 ×10 三个栈（grouped+sdpa、grouped+triton、bmm_dense+triton）全部 0 分歧，且三栈首轮 token 完全一致。**HANDOFF-P6b §2.7 的 shm 竞态根因假设被证实。**
2. **b64 崩溃未复现**：3/3 完成。
3. **attention 决策 = keep_sdpa**（`benchmarks/p6-attention-ab-decision.json`，冻结阈值未动）：TPOT ratio 0.594–0.618（≤0.90✓）、吞吐比 1.48–1.59（≥1.00✓）、CV ≤0.013✓、candidate 自身确定✓，但 **b32/c2048 请求级一致率 0.875 < 0.99**。今晚所有点所有后端自身都确定（含历史不稳定的 sdpa b32）——所以分歧是**可复现的数值差异**（paged 分页/掩码路径），不是运行噪声。转正前置：解释或消除 c2048 分歧，或产品侧接受输出差异。
4. triton_paged 收益量化（供未来）：c2048 吞吐 +48~59%、TPOT -31~41%。

## 3. P2：在线容量 v3（第 3 发加载，`benchmarks/P6-serving-v3.md`）

栈 = sdpa + `decode_moe_backend=bmm_dense`（P0/P1 通过门的组合）；其余与 P5-serving-v2 完全一致。

| 负载档 | v3 SLO% (TPOT p95) | v2 SLO% (TPOT p95) | v3 goodput | v2 goodput | goodput 倍数 |
|---:|---:|---:|---:|---:|---:|
| 50% | **100.0%** (0.229 s) | 100.0% (0.455 s) | 0.427 | 0.411 | 1.04× |
| 75% | **100.0%** (0.348 s) | 26.6% (0.697 s) | 0.656 | 0.163 | 4.02× |
| 90% | **100.0%** (0.312 s) | 11.7% (0.685 s) | 0.706 | 0.077 | 9.17× |
| 100% | **100.0%** (0.401 s) | 7.0% (0.694 s) | 0.771 | 0.050 | **15.42×** |

- TTFT 不再是约束（全档 p95 ≤1.27 s；v2 在 100% 档 2.067 s 越线）。
- 目标「推到 ≥75%」超额：**四档全绿**。512/512 请求成功、全部收到 [DONE]。
- 注意：0.87 QPS 是 v2 口径封顶；「100% 档全绿」说明该口径内无拐点，真实拐点见 §4（P2b）。

## 4. P2b：真实 SLO 拐点探索（第 4 发加载，探索性）

栈同 v3；`--max-qps 1.74`，fractions 0.55/0.625/0.75/1.0（首次尝试 fraction>1 被 `benchmark_online.py` 的断言拒绝，浪费一次加载——正确做法是抬 `--max-qps`）。目标 QPS = 1.74×fraction。

| 目标 QPS | 达成 QPS | 输出 tok/s | TTFT p95 | TPOT p95 | E2E p95 | 最大并发 | SLO% | goodput |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.957 | 0.880 | 56.34 | 1.691 s | 0.535 s | 34.8 s | 39 | **84.4%** | 0.743 |
| 1.087 | 0.943 | 60.32 | 2.847 s | 0.970 s | 63.3 s | 76 | 10.2% | 0.096 |
| 1.305 | 0.944 | 60.40 | 3.921 s | 1.334 s | 85.5 s | 103 | 3.1% | 0.029 |
| 1.740 | 1.117 | 71.47 | 4.226 s | 1.208 s | 77.5 s | 115 | 0.0% | 0.000 |

- **真实拐点在 0.87–0.957 目标 QPS 之间**：0.87 目标（v3 100% 档）仍 100% SLO；0.957 目标跌到 84.4%（TPOT p95 0.535 s 刚过 0.5 s 线）；1.087 起全面崩（TTFT 先崩，队列效应）。
- **硬饱和证据**：目标 1.087 与 1.305 的达成 QPS 钉死在 0.943/0.944、输出吞吐钉死在 60.3/60.4 tok/s——本栈的绝对服务能力上限；1.74 目标时输出 71.5 tok/s 但全部请求违反 SLO。
- **容量结论（v3 + P2b 合并）**：本栈 SLO（TTFT 2 s / TPOT 0.5 s）下可持续负载 ≈ **0.87–0.90 req/s**；绝对吞吐上限 ~60 tok/s。对比 v2 栈同 SLO 下 ~0.41–0.43 req/s → **SLO 容量约 2.1×**。

## 5. P3：图 replay（8B 时间盒 78 分钟 + 30B 验证一发）

| 配置 | 8B 单次 generate | 结论 |
|---|---|---|
| baseline | 0/3（捕获成功后 replay 挂死，10 min watchdog） | 复现率基线 |
| ①replay 前后 rank barrier | 1/3（fail, fail, pass 45 s） | 时间敏感竞态 |
| **①+②每桶私有 pool** | **5/5 全绿**（43–46 s）+ 3 次 generate/清前缀缓存 rc=0 | **候选修复**（历史挂点 1/3 消失） |
| ①+② → 30B | 捕获成功（6 桶 +0.52 GiB）→ **第一次 replay 打死设备**（rank0 watchdog、rank1 死于命令 barrier） | **30B 未修** |

- 8B 修复的两个因素：rank 间 replay 对齐 + 跨桶 graph pool 复用。30B 挂死与之不同源（48 层 × 96 次图内 all-reduce、图内 bmm MoE、更长依赖链）。
- 下一轮可试（未测）：分段捕获（MoE eager）、图内去 MCCL、厂商 MUSAGraph 限制清单。**图方向已按时间盒关闭，`enforce_eager=True` 维持默认。**
- BISECT-1/2 补丁保留在 `model_runner.py`（标注 `# BISECT-1/2`）：8B 有实证收益，eager 默认路径不受影响。

## 6. 明早该做什么（优先级序）

1. **合并/提交本轮代码**（bmm_dense 已是事实上最优 decode 栈；`decode_moe_backend` 透传已进 api_server/p2_benchmark）。
2. **triton_paged 的 c2048 分歧定位**：b32/c2048 上 paged vs sdpa 有 4/32 请求可复现分歧；审查 paged kernel 的页边界/掩码逻辑，或做 bf16 下的 logits diff 上界。通过后重跑 decide --mode attention（门不动）。
3. **30B 图 replay 换路子**：8B 的 barrier+私有 pool 不够；试分段捕获或图内去 MCCL，预算仍按时间盒。
4. **P2b 拐点**（若今夜跑完见 §4）：若 1.5–2.0× 档仍绿，v4 应把 SLO 口径下的目标 QPS 抬到拐点处重测四档。
5. 前缀缓存 cold/hot 探针（P5 遗留）仍值得一次加载补上。

## 7. 证据与 sha256（服务器 /root/ 与 benchmarks/data/p6b/ 各一份）

| 文件 | 内容 | sha256（前 16 位） |
|---|---|---|
| p6_eager_matrix.json/.jsonl | P0 矩阵 | cdca82dd9739886d / e4106472b625445e |
| p6_d2_nightly.json/.jsonl | P1 合并 D2 | 30dd1e646a71b9f8 / aa0d19b5b1b7628c |
| p6_attention_ab.json | P1 attention A/B | 26e2d457eacca22c |
| p6-attention-ab-decision.json | 冻结门决策 | 8bfd072c4ed854de |
| online-v3.json | P2 在线 v3 | 84b8ac7a44b8eda4 |
| online-v3-highload.json | P2b 高负载四档（真实拐点 0.87–0.957 QPS） | 9f1c8ea91ec4969aa5906c40fccf753512052b77a99948e2b1a53cac7f7ad8b8 |
| p6_graph_5x.log / p6_graph_bisect1.log / p6_graph_bisect2.log / p6_graph_3gen_bisect12.log | P3 复现率与修复证据 | 见文件 |
| p5_graph_ab.log | 30B 图 A/B（捕获成功 + replay 失败现场） | 见文件 |
| P6-serving-v3.md / P6-nightly.md | 报告 | d1d65309ed07e7a9 / （本文件） |
