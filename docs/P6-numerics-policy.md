# 数值契约（numerics policy）— attention 后端的等价性标准

> 建立于 2026-10-27，起因：I4（triton_paged vs sdpa）长期被"≥99% token 一致"卡住，而该阈值**严于业界实践**。
> 本文档定义：不同 attention 后端之间**什么算等价**、**怎么验收**、**基线怎么维护**、**出问题怎么回退**。
> 相关：`docs/ISSUES.md` I4、`docs/P6-I4-attention-logits.md`（诊断）、`scripts/p6_decide_attention_equivalence.py`（门实现）。

---

## 1. 背景：为什么两个后端不可能逐 token 相同

两个后端算的是**同一个数学函数**，差别只在浮点归约顺序：

| | SDPA 路径（`attention.py::_forward_decode`） | Triton paged（`kernels/paged_attention.py`） |
|---|---|---|
| KV 取用 | `k_cache[blocks]` **gather 成稠密张量**，再做 GQA 扩展、`where` 掩码零化 | 按页表逐页读，**不物化**、不扩展、不建 mask |
| 归约 | 厂商 flash 内核的 block 划分与 fp32 累加次序 | online softmax，`BLOCK_N=32` 逐块重标定 `m/l` |

IEEE-754 浮点加法不满足结合律，`(a+b)+c ≠ a+(b+c)`，所以两者末位必然不同。bf16 只有约 3 位十进制有效位，跨 48 层放大后表现为：

- 首层输出最大差 `2.4414e-4`；
- 第 36 层 logits 差 `8.72`（最终 `6.09`）；
- 8,928 个 request-step 中 **19 次 top-1 翻转**（0.21%），**全部**发生在 baseline top-2 margin ≤0.3125 的近平局。

**这不是 bug**，已排除的部分：页边界矩阵（255/256/257、511/512/513、2047/2048/2049）、乱序页、poison tail 全部通过（`max_abs=1.2207e-4 < 2e-3`）；没有 mask / tail / 页地址错误；两个后端**各自都是自确定的**。

## 2. 业界实践（为什么改门而不是改内核）

| | vLLM | SGLang | DeepSeek DeepInfer |
|---|---|---|---|
| 默认可复现 | **不保证**（为性能让步） | 不保证；文档明确 batch size 变化会改变归约切分，**`temperature=0` 也会变** | 做成 opt-in 一等特性 |
| 确定性开关 | `VLLM_BATCH_INVARIANT=1`（beta） | `--enable-deterministic-inference` | `DEEPINFER_BATCH_INVARIANT=1` + `DEEPINFER_ENABLE_MTP_VARLEN=0` + 固定 seed |
| 后端差异 | 文档列为差异来源之一；只保证"同硬件+同版本+同后端" | 换后端**必须重新做数值验证** | 承认 FP4/FP8+融合会改变累加顺序 |
| 验收方式 | 容差 / 精度基准 | 精度基准（GSM8K 196/200 vs 198/200 也算通过） | L1 结构检查 + **L2 teacher-forced 逐元素 logprob** + 上线前 ~100 条真实 prompt 一致性评估 |

**结论：没有任何一家承诺跨后端逐 token 一致。** 通行做法是「固定一个后端 + 容差/质量等价校验 + 保留参考后端对拍」。我们的旧门（≥99% token 一致）比业界标准严，因此按本文档替换。

## 3. 我们的三层等价门（替换 I4 的单一 token 门）

> 旧门（`p5_decide.py::ATTENTION_GATES`）**不改**，保留为历史决策依据（其结论 `keep_sdpa` 是当时的正确判断）。新门由 `scripts/p6_decide_attention_equivalence.py` 实现。

| 层 | 检查 | 阈值（冻结） | 现状 |
|---|---|---|---|
| **L1 结构正确性** | 页边界/乱序页/poison-tail 参考误差 | `max_abs < 2e-3` | ✅ 通过（1.2207e-4） |
| **L2 数值等价** | 同状态 teacher-forced：top-1 翻转率 + 翻转处 margin | 翻转率 **< 0.5%**；**全部**翻转处 top-2 margin < 0.5 | ✅ 0.21%，margin ≤0.3125 |
| **L3 输出质量** | 100 条真实 prompt 双后端对拍：首 token 一致率、退化检查、**LCP 分布**（只报告）、**12 组极端/中位样本人工判读** | 首 token 一致率 **≥99%**；无退化；**判读样本 100% 语义等价且 0 退化** | ✅ 通过（首 token 100%、退化 0/0、判读 12/12 等价） |
| **L4 自确定性** | 每后端 3 次重复 | 逐 token 一致 | ✅ 已验（b32 ×10 三栈零分歧） |
| **L5 性能** | 在线四档 SLO | 100% 档 SLO 不劣化 | ⏳ 待跑 v4 |

**L3 的指标演进（一次被数据推翻的设计，如实记录）**：初版门用**字符级编辑距离 p95 ≤ 8%**。实测 100 条 prompt 后该指标**严重不过门**（p50 23.3%、p95 65.1%、仅 20/100 条完全一致）——但随即发现这是**指标选错**而非质量下降：

| 观测 | 数值 | 含义 |
|---|---:|---|
| 首 token 一致率 | **100%** | 第一步 argmax 几乎总相同（数值差 < margin） |
| **LCP 中位 / 均值** | **42.5 / 60.7** token（共 128） | 前 1/3~1/2 完全一致，之后一次近平局翻转 → 后续走上不同分支 |
| token 平均一致率 | 0.509 | 与"翻转后整段不同"自洽（≈65/128） |
| 12 组人工判读 | **12/12 语义等价，0 退化** | 差异全是措辞/框架选择（例："降低显存/提高速度/降低功耗" vs "计算量/内存/延迟"） |

**结论**：一旦在某个近平局翻转，后续文本必然整体不同——**编辑距离度量的是"措辞"，不是"质量"**。业界（DeepInfer）明确建议用"真实 prompt 人工/任务级一致性评估"而非 token 距离。因此 L3 定为：首 token + 退化检查 + **人工判读样本**（本次 12 组，见 `benchmarks/P6-L3-review.md`）+ LCP 只报告。

**L3 的已知缺口（诚实记录）**：计划中的"交叉困惑度"需要**多位置 logits**，而引擎目前只回 last-token logits（`embed_head.py:58-60`，与 S2 同一处改动）。交叉 PPL 作为后续项，落地后补测——那时可以把 L3 的人工判读从样本扩大成全量自动打分。

## 4. 基线（per-backend）规则

1. **每个后端各自冻结一份参考**：`benchmarks/p2-reference-8tok.json` 只对 `sdpa` 有效；paged 转正后需生成 `p2-reference-8tok-paged.json` 并单独冻结。
2. `scripts/p2_benchmark.py --verify-backends` 的行为不变（它是 MoE 后端门）；但 **attention 后端切换后不得再用 sdpa 参考做硬门**，否则必然误报。
3. 跨后端 A/B 只做**报告**，不做硬门；硬门用本文档 §3 的五层。
4. 任何后端升级（改 kernel、换厂商库版本）都必须重跑 L1–L4。

## 5. 回退

| 触发 | 动作 |
|---|---|
| L3 或 L5 不达 | 默认回退 `sdpa`，paged 保留 `--attention-backend triton_paged` opt-in |
| 线上出现语义退化 | 同一条：切回 sdpa（行为差异只来自归约顺序，切换本身无状态） |
| 后续要做 batch-invariant | 我们**拥有**该 kernel（自研 Triton），可直接固定归约顺序；这是 vLLM/SGLang 用厂商内核时做不到的（它们在部分硬件上只能"选后端"） |

## 6. 记录

- 决策 JSON：`benchmarks/p6-attention-equivalence-decision.json`
- 数据：`benchmarks/data/p6b/p6_attention_equivalence.json`
- 旧决策（历史）：`benchmarks/p6-attention-ab-decision.json`（`keep_sdpa`，基于 token 一致率门）
