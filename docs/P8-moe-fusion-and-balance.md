# P8 — 双卡 MoE：融合、容量收缩与负载均衡技术方案

> 日期：2026-10-27。配套：[`P8-architecture-review.md`](./P8-architecture-review.md) §3、[`P8-24h-plan.md`](./P8-24h-plan.md) P3。
> 目标：回答"双卡 MoE 是否充分用上了现有优化、要不要 MegaMoe、负载是否均衡、哪些必须手搓"。
> 规则：所有收益数字分两类——**实测**（带证据）与**推导估计**（标明假设）；推导值必须被 24h 内的微基准证实或推翻。

---

## 0. 结论

1. **拓扑是对的**：EP2、每 rank 64 个完整 expert、hidden 复制、每层一次 all-reduce。TP2 下 all-to-all 要搬 `[T×8, 2048]`，比 all-reduce 的 `[T, 2048]` 贵 8×；**不要上 DeepEP/all-to-all**。
2. **"负载均衡 5%"是被口径掩盖的假象**：`grouped_mm`/`bmm_dense` 把非本 rank 的 slot 折叠到 local expert 0 并**真的计算**。按折叠口径，expert 0 拿到约 **4.06·T** 个 slot，其他 expert 约 **0.0625·T**，**内部倾斜约 65:1**；同时 **50% 的 slot 是零权重的无效行**。
3. **最高 ROI 不是 MegaMoe，而是 `bmm_dense` 的容量收缩**：现在每个 local expert 都按 `S = T×K` 的最坏容量做 dense bmm（b32 时每层 64×256 行，其中有效行约 4T/64）。改成"只打包 owned slot + `S = counts.max()`"后，按实测 bmm 时间拟合（`t ≈ 1.22 ms + 0.0064 ms/行`）估算，b32 每层 2.86 → ~1.3 ms，b64 5.17 → ~1.3 ms。**推导估计：decode MoE 层 −50~75%，30B 步时 −25~40%**。这是本轮唯一可能改变容量曲线的单点。
4. **第二优先是 fused route/pack**：route/pack 实测 558.8 ms，与两次 GEMM（700.4 / 562.0 ms）同量级；融合成 2–3 个 Triton kernel（histogram → offsets → scatter+gather）可去掉 `argsort` 与多次 `index_select`，并能顺手过滤非 owned slot。
5. **MegaMoe 只能搬子集**：FP8 计算、TMA、all-to-all 在 S4000/PCIe 上不成立；可搬的是 dispatch/activation/combine 融合。不要按"MegaMoe 复刻"立项。
6. **厂商生态的边界**：`fused_experts` 可复用但默认慢 2.3–6.3×、mp22 数值错误（不可用）；`torch._grouped_mm` 无旋钮；能调的只有我们自己的 Triton kernel 与 vLLM 的 autotune 配置文件（E2）。

---

## 1. 拓扑与代价模型

### 1.1 布局（实测/代码）

| 项 | 值 | 证据 |
|---|---|---|
| layers | 48 | `tests/p6_moe_memory_probe.py:33` |
| hidden | 2048 | 同上 |
| experts / top-k | 128 / 8 | `README.md:26` |
| experts per rank | 64 | `qwen3_moe.py:86-90` |
| moe_intermediate | 768 | `tests/p6_moe_memory_probe.py:32` |
| expert weights | gate_up `[64,2048,1536]` + down `[64,768,2048]`，bf16 | `qwen3_moe.py:115-127` |
| 每 rank expert 权重 | 576 MiB/layer × 48 = 27 GiB | 同上；`HANDOFF-P6b.md` §2.3 |
| hidden 复制 | attention `o_proj` all-reduce 后每 rank 完整 | `linear.py:152-156` |
| 每层通信 | EP all-reduce `[T,2048]` + attention all-reduce `[T,2048]` | `qwen3_moe.py:386-387` |
| 每步通信 | 96 次 all-reduce + 1 次 embedding all-reduce + 1 次 logits gather | `embed_head.py:34-42, 56-66` |

### 1.2 代价

- 孤立 MCCL all-reduce 0.076–0.103 ms/次 → 96 次 **7.5–9.9 ms**（`HANDOFF-P6b.md` §2.4）。
- 模型内 EP all-reduce 占 forward **2.48–5.45%**；b32/c2048 是 **393.0 ms / 7915.5 ms**（`P5-profiling.md:35-44`）。
- 393 ms ≫ 10 ms 的带宽下限 → **393 ms 主要是在等另一 rank**（straggler / 路由偏斜），不是通信带宽。
- 结论：优化通信带宽没有意义；**优化 rank 间完成时间差**才有意义。

---

## 2. 折叠路由的真实代价（本轮最重要的代码发现）

### 2.1 代码事实

`grouped_mm`（`qwen3_moe.py:213-256`）：

```python
local_raw = global_to_local[topk_ids]      # 非本 rank = -1
owned = local_raw >= 0
local_ids = local_raw.clamp(min=0)         # -1 → 0 （折叠到 local expert 0）
weights = topk_weights * owned             # 非 owned 权重 = 0
...
counts = ...index_add_(local_ids)          # expert 0 收到所有非 owned slot
offsets = counts.cumsum(0)
tokens = hidden.index_select(0, sorted_tokens)
torch._grouped_mm(tokens, gate_up_proj, offs=offsets)
```

`bmm_dense`（`qwen3_moe.py:292-326`）做同样的折叠，只是后面用 `S = T×K` 的固定容量 dense bmm。

### 2.2 数学后果

EP=2、K=8 时，每个 token 平均 **4 个 owned slot、4 个 non-owned slot**：

| 量 | 表达式 | b32 时 |
|---|---|---:|
| 每层本地 slot 总数 | `T×K` | 256 |
| 有效（owned） | `T×K/2` | 128 |
| 无效（zero-weight） | `T×K/2` | 128 |
| expert 0 的折叠负载 | `≈T×K/2 + T×K/(2×64)` | ≈130 |
| 其他 expert 平均 | `T×K/(2×64)` | ≈2 |
| **内部倾斜** | **≈65:1** | — |

两个直接结论：

1. **`grouped_mm` 有 50% 的行是零权重**，GEMM 仍要算。
2. **expert 0 是层内 straggler**（65× 平均），grouped GEMM 的并行度被它拖住。

### 2.3 为什么现有指标没发现

- `local_pairs` 只在 opt-in profiler 里统计 `owned` 的 slot（`qwen3_moe.py:240-246`）；
- P5 的 rank 间不平衡 1.69–5.19% 是**跨 rank**、且只算 owned；
- `collect_expert_stats` 统计的是**全局** `topk_ids` 的 bincount（`qwen3_moe.py:366-369`），两个 rank 相同，看不到"折叠到 expert 0"的本地倾斜。

所以"负载均衡 5%"既不是 intra-rank 的真相，也不是 folding 的真相。

### 2.4 测量方案（24h P3.1）

1. 打开 `collect_expert_stats`，用 32 条真实 chat prompt（中英各 16），prefill 512 / decode 64，记录每层全局 `expert_counts`。
2. 新增一次性统计（不改主路径）：按 `global_to_local` 计算每层每 rank 的 `owned_counts[e]` 与 `folded_extra`（非 owned 但 local_id=0 的数量）。
3. **计数点必须在 fold 之前**：在 `topk_ids` 处（`qwen3_moe.py:215-221` 之前）插桩，统计真实路由；`local_pairs` 只含 owned slot，**看不到 expert 0 的折叠倾斜，不可用于本项**。先确认现有 `expert_counts` 是在 fold 前还是 fold 后累加的——若在 fold 后，本项全部作废重测。
4. 输出：`expert_load_max/mean`、Gini、`rank_imbalance`、`expert0_ratio`、`waste_ratio = 1 - owned/total`。

**这是所有后续决策的前提**：如果真实路由的 `counts.max()` 接近 `T×K`（极端偏斜），容量收缩的收益会消失；如果接近平均值（正常），收益成立。

---

## 3. bmm_dense 容量收缩（最高 ROI）

### 3.1 现状

`bmm_dense` 为每个 local expert 分配 `S = T×K` 行（`qwen3_moe.py:275-288, 313-316`）：

```python
x_pad = zeros([E_local, S, H])          # S = T*K
x_pad[sorted_ids, ranks] = x_slots
expert_out = bmm(act(bmm(x_pad, gate_up)), down)
out = expert_out[sorted_ids, ranks] * weights
```

- b32：`S=256`，`x_pad [64,256,2048]` bf16 = 64 MiB/layer；
- 但每个 expert 实际只需要 `counts[e]` 行，平均 ~4 行、最大 ~8–16 行（待测）。
- **有效行占比约 1/64 ~ 1/128**（b32）。

### 3.2 实测时间拟合

`HANDOFF-P6b.md` §2.3 的 bmm_dense 每层时间：

| batch | S=T×K | ms/layer |
|---:|---:|---:|
| 1 | 8 | 1.268 |
| 2 | 16 | 1.299 |
| 4 | 32 | 1.375 |
| 8 | 64 | 1.944 |
| 16 | 128 | 2.301 |
| 32 | 256 | 2.858 |
| 64 | 512 | 5.169 |

线性拟合（b1/b32 两点）：`t ≈ 1.22 ms + 0.0064 ms/行`。**固定开销 ~1.2 ms/layer**，边际成本很小。

### 3.3 设计：owned-only + 动态 S

```
1) route：只打包 owned (token, slot) 对（过滤 -1），得到
   sorted_token/local_id/rank/weight，长度 n_owned ≈ T×K/2
2) counts[e] = histogram(local_id)         # 设备端，整数加法，确定性
3) S = counts.max().item()                  # 每层 1 次 host sync
4) x_pad = zeros([E_local, S, H]); x_pad[local_id, rank] = hidden[token]
5) bmm ×2；gather expert_out[local_id, rank] * weight
6) 按原 slot 顺序 combine（沿用现有 deterministic 路径）
```

**数值**：每行的 K 维归约不变；M 维大小不影响单行点积顺序 → 理论逐 token 全等（仍需实测确认 kernel tiling 不改 K 顺序）。

**预期**（推导，待实测）：

| batch | 现状 S | 估计 S_max | 现状 ms | 估计 ms | 降幅 |
|---:|---:|---:|---:|---:|---:|
| 32 | 256 | 8–16 | 2.86 | ~1.3 | **55%** |
| 64 | 512 | 16–32 | 5.17 | ~1.3–1.5 | **70–75%** |

若 MoE 占 decode 步 ~50%，则 b32/b64 步时 **−25~35%**；对应 b64 从 141 tok/s 量级向 **190–220 tok/s** 走（乐观上界，需实测）。

### 3.4 风险与对策

| 风险 | 对策 |
|---|---|
| `S_max` 仍很大（路由极端偏斜） | P3.1 先测；若 `S_max > 2×均值` 则收益打折，仍按实测判决 |
| 每层 `counts.max().item()` 的 sync（48 次/步） | 实测开销；若 >2 ms，改为固定容量 C + 溢出 expert 回退 grouped_mm（或 graph 分桶） |
| 小 M bmm 效率下降 | 实测；固定开销 1.2 ms 是下限，收益上限受它限制 |
| 破坏图的单形状假设 | 图本来未启用；未来按 S 分桶捕获（4/8/16/32/64） |
| 数值不一致 | 门：先试 32 条 prompt × 128 token 与现行 bmm_dense **逐 token 全等**（bonus，不是硬门）；若 bmm 形状从 GEMM 退化为 GEMV 导致归约顺序变化，则走 L1 `max_abs ≤2e-3` + L2 翻转率 ≤0.5% + 自确定 3/3——这与 bmm_dense 当初采用的标准一致，不是降级 |

### 3.5 判决门（先写后跑）

- 正确性（**不要求与现行 bmm_dense 逐 token 全等**）：自确定 3/3（同一输入三次 token 一致）+ L1 `max_abs ≤2e-3` + L2 翻转率 ≤0.5%、翻转处 margin <0.5；逐 token 全等只作为 bonus 报告。
- 性能：b32 与 b64 的 **MoE 层时间 −≥20%**；decode 步时 **−≥10%**；交错重复 ≥5 次、CV ≤5%。
- 显存：`x_pad` 峰值不高于现状（应显著更低）。
- 不过门：保留为 opt-in，不设默认。

**这是 24h 内唯一有潜力改变容量曲线的 MoE 改动**，优先级高于 fused_grouped_mm。

---

## 4. fused route/pack（第二优先）

### 4.1 现状成本

P5 profile（b32/c2048，8 个 decode step × 48 层，grouped_mm）：

| 段 | 时间 | 每层折算 |
|---|---:|---:|
| route/pack | 558.8 ms | 1.45 ms |
| grouped GEMM 1 | 700.4 ms | 1.82 ms |
| grouped GEMM 2 | 562.0 ms | 1.46 ms |
| combine | 22.6 ms | 0.06 ms |

route/pack 由 `argsort`、`arange`、`index_select`、`index_add_`、`cumsum` 等 8 个左右的独立 op 组成（`qwen3_moe.py:222-239, 296-312`），**每个 op 都是一次 launch**；测到的时间含 launch 开销（P5 自己也注明短 segment 的 event 开销可能夸大）。

### 4.2 设计

用 2–3 个 Triton kernel 替代：

1. **hist**：每个 program 处理一段 slot，`atomic_add(counts[expert], 1)`；整数加法与顺序无关 → 确定性。
2. **scan**：E=64，单 program 前缀和（或复用 `torch.cumsum`，成本可忽略）。
3. **scatter+gather**：每个 slot 计算 `pos = atomic_add(cursor[expert], 1)`，把 `hidden[token]` 行直接写到 `tokens[pos]`，同时写 `sorted_slot[pos]`、`weight[pos]`。行内顺序变化不影响每行 GEMM 结果的数值（行独立），combine 仍按原 slot 顺序归约 → 最终确定性。
4. 非 owned slot 在 hist/scatter 阶段直接跳过 → 同时解决 §2 的 50% 无效行。

预期：route/pack 1.45 → 0.5–0.8 ms/层；加上过滤，grouped GEMM 行数减半 → MoE 段 −20~35%（推导，需微基准）。

### 4.3 门

- 微基准：route/pack 段 **−≥30%**；端到端 MoE 段 −≥5%。
- 正确性：tiny + 30B 各 32 prompt × 128 token 逐 token 全等（首选）；否则 L1/L2 + 确定性。
- 实现范围：24h 只做到"微基准 + 设计定稿"；改引擎主路径放到下一轮（除非 P3 时间富余）。

---

## 5. 负载均衡与 expert 放置

### 5.1 已有机制

| 机制 | 实现 | 状态 |
|---|---|---|
| `linear` | rank0 0-63，rank1 64-127 | 默认 |
| `round_robin` | 奇偶 | 已在 tiny 测过 |
| `profiled` | LPT 贪心 + 模型 hash 校验 | 已实现，**真实负载未跑** |
| 统计 | `collect_expert_stats`（全局 bincount） | 默认关 |
| 生成器 | `scripts/profile_experts.py`（batch 32、prompt 128、output 64、随机 token） | 已有 |

证据：`qwen3_moe.py:138-160`、`scripts/profile_experts.py:9-86`、`config.py:34, 39`。

### 5.2 为什么之前判"不需要"

P5 的 rank 间不平衡最大 5.19%（b32/c2048），低于预冻结 10% 门 → 否决 profiled。问题：

- 用的是 synthetic 随机 token；
- 只算 owned slot；
- 没有考虑折叠造成的 expert 0 倾斜；
- 没有在 `bmm_dense` 下重测（现在 decode 已换后端）。

### 5.3 24h 测量与 A/B

1. 真实 prompt 统计（§2.4），输出 `S_max`、`expert0_ratio`、`waste_ratio`、rank imbalance（折叠修正口径）。
2. 用真实负载生成 `profiled` map，跑 `linear` vs `profiled`：
   - b32 decode（128 token）×3；
   - b32 prefill（512）×3；
   - 记录吞吐、CV、每 rank 的 `S_max`。
3. 门：**吞吐 ≥+5%** 或 **rank 不平衡 −≥10%**；否则关闭，写"不采用"。

### 5.4 与容量收缩的协同

容量收缩后每层时间由 `S_max` 决定，**放置直接影响 `S_max`**：把热点 expert 均分到两 rank 可降低两边的 max。因此"先测真实负载 → 再决定 profiled"的顺序不能反。

---

## 6. MegaMoe 可移植性（结论表）

| MegaMoe 技术 | 硬件/软件前提 | S4000 现状 | 可移植 | 替代 |
|---|---|---|---|---|
| FP8 expert 计算 | FP8 tensor core | 无 FP8 compute；权重加载反量化 bf16（`loader.py:28-35`） | ❌ | bf16 |
| TMA / warp-specialized 拷贝 | SM90+ | 无 | ❌ | 普通 Triton |
| persistent MoE kernel | 大 shared mem + cluster | 不确定；Triton mtgpu 限制 | 部分 | 分段融合 |
| all-to-all（DeepEP） | NVLink / 高带宽 | PCIe Gen5、无 NVLink | ❌ | hidden 复制 + all-reduce |
| fused dispatch/combine | 自定义 kernel | 可写 Triton | ✅ | `moe_fusion.py` + 新 fused route |
| shared experts | 模型结构 | Qwen3-30B-A3B 无 | N/A | — |
| 图内 MoE | 可捕获 kernel | bmm 可捕获；replay 漂移 | 待定 | 先修图（P2） |

**判断**：按"可移植子集"立项——**容量收缩 > fused route/pack > 其余不做**。

---

## 7. 通信：为什么不做 all-to-all / overlap

- **all-to-all 不划算**：TP2 下 all-reduce 搬 `[T,2048]` bf16（4 KB/token），all-to-all 要搬 `[T×8,2048]`（32 KB/token），贵 8×；而且当前 393 ms 的 in-model 时间主要是等 straggler，不是带宽。
- **overlap 不划算**：EP all-reduce 占 forward 2.48–5.45%，低于预冻结 20% 门（`P5-moe-optimization-survey.md:56-59`）。先解决偏斜，再谈重叠。
- **bucketing**：每层 2 次 all-reduce（attention + MoE），已在不同位置；合并会改变依赖链，收益不明，不做。

---

## 8. 分期路线与门

| 期 | 内容 | 成本 | 门 | 产出 |
|---|---|---|---:|---|---|
| M0 | 真实负载统计（§2.4，**fold 前计数**） | 0.5 h | 每层 counts、S_max、waste/ratio | `p8_moe_balance.json` |
| M1 | **bmm_dense 容量收缩原型**（§3） | 2 h | 自确定 3/3 + L2 软门；MoE −20%；步时 −10% | `p8_moe_capacity.json` |
| M2 | `linear` vs `profiled` A/B（容量收缩开启） | 1.5 h | 吞吐 +5% 或 imbalance −10% | 采用/关闭 |
| M3 | fused route/pack 微基准 | 1.5 h | route −30%；MoE −5% | `p8_moe_fusion.json` + 设计 |
| M4 | 改引擎主路径（下一轮） | 4–6 h | M1/M2/M3 门 + 回归 | 默认切换 |
| M5 | KV 量化（I20，周级） | — | L1–L4 + 容量/带宽 | 另立 |

**为什么 M1 在 M2 之前**：容量收缩是 30B 唯一可能改变容量曲线的单点（估算 decode MoE 层 −50~75%），且不需要真实负载统计就能实现；placement 的 +5% 门在容量收缩后才有意义（否则瓶颈仍在 dense bmm 的行数上，放置差异被掩盖）。

**硬约束**：M1/M3 都只在 `decode_moe_backend` 上生效；不改 `moe_backend`（prefill）默认；不做全局 `bmm_dense`（prefill OOM）。

---

## 9. 参考

- 代码：`nanovllm/models/qwen3_moe.py`、`nanovllm/kernels/moe_fusion.py`、`nanovllm/kernels/vendor_moe.py`、`scripts/profile_experts.py`、`scripts/p6_moe_microbench.py`（若存在）
- 报告：`benchmarks/P5-profiling.md`、`docs/P5-moe-optimization-survey.md`、`docs/HANDOFF-P6b.md` §2.3–2.4、`benchmarks/P6-nightly.md` §1
- 上游：vLLM `fused_experts` / `VLLM_TUNED_CONFIG_FOLDER`、SGLang `fused_moe`、MegaMoe（FP8+TMA+persistent）、DeepEP（all-to-all）
- 结论：**买得到的只有 vendor `fused_experts`（默认慢、mp22 错），其余必须用 Triton 手搓；但手搓的顺序应是容量收缩 → route/pack，而不是复刻 MegaMoe。**
