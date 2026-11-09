# P29 最终收口：探索结论与面试口径

> 日期：2026-11-09。Day1–Day6 已执行；本文件是收口/面试材料，不是新的产品承诺。
> 生产默认仍是：TP2/EP2、`triton_paged`、`grouped_mm` prefill、`bmm_dense` decode、eager、`stall_free`、chunk 256。

## 1. 一页结论

| 方向 | 结论 | 关键证据 |
|---|---|---|
| TP2+EP2 | **保留生产默认** | 30B bf16 56.9GiB 放不进单卡；TP2 后 2 KV 头/rank；EP2 64 专家/rank；TP+EP 用 all-reduce（DP=1） |
| PP/SP/CP | **不做** | 2 卡 PP 有气泡；4k 上下文下 SP 激活不是瓶颈；CP 需要 32k+ |
| DP+EP / 30B DP2 | **NO-GO** | 0.6B 架构原型 1.20×，但 30B 需要 FP8 保留；naive dequant 738ms/forward > 142ms step |
| 8-bit / 4-bit | **不做 e2e** | ATen INT8/W8 慢 4–57×；自定义 W8 GEMV 0.76×、GEMM 0.06–0.26×；MATE 不可用 |
| FP8 保留 | **NO-GO** | Triton FP8 dequant 编译失败；`_scaled_mm` 不支持；naive dequant 太慢 |
| round-robin 放置 | **NO-GO** | imbalance 8.5%→0.3–0.5%，但 30B token 门 FAIL（第 4 条从首 token 分叉） |
| 算子融合 | **局部保留，不继续堆** | fused SiLU/combine 存在；fused RMSNorm A/B 仅 −0.8%；bmm 容量 −10% 门 FAIL；图被 MCCL 阻塞 |
| overlap 调度 | **已有 opt-in，不重跑** | P23：bitwise 正确、b1 0.9972×、b16 0.9978×，没有步时收益 |

## 2. 分布式并行：面试怎么讲

### 2.1 为什么是 TP2+EP2

- 30B bf16 权重约 **56.9GiB**，单卡 48GB 放不下；TP2 每 rank 约 29.5GiB packed。
- 48 层、4 个 KV 头：TP2 后 2 KV 头/rank；TP4 只剩 1 个，all-reduce 更重，不划算。
- EP2：128 专家 / 2 = 64 个完整专家/rank；router 复制；本地专家 partial sum 后一次 all-reduce。
- TP+EP（DP=1）是低延迟形态，通信是 **AllReduce**；DP+EP 才是 AllToAll/高吞吐。
- PP 在 2 卡上会引入气泡；SP/CP 在当前 4k 上下文下不是主要矛盾。

### 2.2 当前实测

- prefill EP imbalance：**8.4–8.9%**；decode：**4.2–5.6%**（Day1，补 `bmm_dense` 观测后）。
- EP all-reduce 占 forward：prefill 2.6–7.8%，decode 2.9–3.9%。
- 30B decode clean step：**163 / 184 / 243 ms**（Day1 harness b1/b8/b32；历史生产 b1 约 142ms，口径不同不要混用）。
- RR 放置能把 imbalance 压到 0.3–2.5%，但 30B token 门 FAIL；所以不是“没做”，是**做了但过不了正确性门**。

## 3. 算子/量化：面试怎么讲

### 3.1 已经做实的

- 自研 Triton **paged decode attention**；
- `grouped_mm` prefill + `bmm_dense` decode；grouped 比 loop 快 3.0–3.2×，bmm 比 grouped 快 1.44–1.53×（b16）；
- 确定性 top-k combine（修掉 `index_add_` 原子累加导致的贪心不可复现）；
- fused SiLU+mul / weighted combine 存在，但只是 opt-in；
- KV int8 三方案都试过，质量门 FAIL，未默认。

### 3.2 量化为什么没做成

- FP8 权重只在加载期反量化；S4000 规格无 FP8，`_scaled_mm` 直接报 `MUSA_ARCH >= 310`；
- ATen `_int_mm` / `_weight_int8pack_mm` 在 2048² 下比 bf16 慢 **4–57×**；
- 自定义 W8 GEMV 0.76× bf16；W8A16 GEMM 0.06–0.26× bf16；
- FP8 保留 + naive dequant：**738 ms/forward**，当前 step 142ms；
- Triton 内 FP8 dequant 编译失败；MATE 未安装；所以不是“没试 4-bit”，而是**8-bit 都没有可用的速度路径**。

### 3.3 量化的真实价值

- 显存：checkpoint 29.03GiB（FP8 27.84 + bf16 1.19）；bf16 56.88GiB；FP8/W8 完整模型约 29GiB，单卡可放；
- 但 30B DP2 还需要 FP8 保留 + 单卡完整模型；当前 loader 不保留 FP8，naive dequant 太慢；
- 所以量化目前只能作为“内存模型/未来方向”，不能当 decode 加速或 30B DP2 的现成方案。

## 4. 0.6B DP2 原型怎么讲

- 两个独立 TP1 副本 vs 一个 TP2：TP2 median 232.75 tok/s，DP2 279.13 tok/s，**1.20×**，CV 1.6%/2.7%；
- 设备隔离 `MUSA_VISIBLE_DEVICES` 可用；
- 但这是 0.6B dense，不是 30B MoE；**不能外推**；30B DP2 需要 FP8 保留，Day6 已判 NO-GO。

## 5. 面试 Q&A（可直接背）

**Q1：为什么不用 PP/SP/CP？**
> 2 卡、30B 模型，TP2 已经放得下；PP 会引入 pipeline 气泡，PP+EP 组合限制多；SP 在当前 4k 上下文下激活不是主瓶颈；CP 是 32k+ 长上下文才需要。不是不会，是 2 卡上 ROI 不成立。

**Q2：EP 为什么用 AllReduce，不用 AllToAll？**
> 我们是 TP+EP、DP=1，hidden 复制、专家切分，本地专家 partial sum 后 all-reduce 合并；这是 vLLM MoE playbook 里 TP+EP 的低延迟形态。AllToAll 是 DP+EP 的高吞吐形态，需要 DP Attention 和跨卡 dispatch/combine，2 卡上不划算。

**Q3：EP 负载不均衡怎么办？**
> 我们先量到 prefill imbalance 8.4–8.9%、decode 4.2–5.6%；试了 round-robin，imbalance 降到 0.3–2.5%，但 30B token 门 FAIL（第 4 条从首 token 分叉），所以没有转正。下一步如果要继续，必须先做逐专家权重审计 + 首 token logits margin，区分数值敏感性和装载 bug。

**Q4：算子融合做到什么程度？**
> paged decode、grouped_mm/bmm_dense、确定性 combine 是自研；fused SiLU/combine 存在但 opt-in；fused RMSNorm A/B 只 −0.8%，被否决；bmm 容量收缩 −10% 门 FAIL；整图被 MCCL 楔死，分段图在线 A/B FAIL。结论是：融合是局部有、系统级没有，当前瓶颈不是单个 elementwise kernel。

**Q5：8-bit/4-bit 能做吗？**
> 8-bit 在 S4000 上没有可用速度路径：ATen int8 慢 4–57×，自定义 W8 GEMV 0.76×、GEMM 0.06–0.26×；FP8 运行时直接不支持；FP8 保留 naive dequant 738ms/forward。4-bit 更靠后。量化现在只能拿显存收益，不能宣称 decode 加速。

**Q6：如果给你更多时间，做什么？**
> 优先级不是继续调 kernel，而是：① 异步调度/多副本运维能力（不碰数值路径）；② 长上下文/KV 容量；③ 只有拿到 MATE/C++ 支持才做 FP8/DP2。RR/4-bit/PP 已经被数据否掉，不会为了填时间重做。

## 6. 简历 bullet（保守版）

> 在 2×MTT S4000 上把 Qwen3-30B-A3B（128 专家 top-8）做到生产可用的 OpenAI 兼容推理服务：TP2+EP2、paged attention、grouped/bmm MoE 后端、确定性 combine、stall-free 调度；30B b1 decode 约 142ms、b64 离线约 141.5 tok/s；建立预注册门 + 16 套回归 + 产品化部署。另完成 DP/RR/8-bit/FP8 保留的系统性评估，用实测否掉不可行路径，避免 negative result 被包装成 feature。

## 7. 不要声称的

- 不要说“实现了 PP/SP/CP/DP+EP 全套”；没有。
- 不要说“4-bit/8-bit 加速”；没有速度收益。
- 不要说“30B DP2 已落地”；只有 0.6B 架构原型，30B NO-GO。
- 不要说“round-robin 提升了吞吐”；只降低了 imbalance，且 token 门 FAIL。

## 8. 证据索引

- [Day1](../benchmarks/P29-day1.md)：平台/拓扑/EP imbalance
- [Day2](../benchmarks/P29-day2.md)：FP8/W8 内存与 GEMV
- [Day3](../benchmarks/P29-day3.md)：MATE/W8 GEMM/RR 放置
- [Day4](../benchmarks/P29-day4.md)：RR token 门 + A/B
- [Day5](../benchmarks/P29-day5.md)：0.6B DP2 vs TP2
- [Day6](../benchmarks/P29-day6.md)：FP8 保留审计 / 738ms
- 总路线：[ROADMAP-NEXT.md](ROADMAP-NEXT.md)