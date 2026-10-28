# P8 — MoE:真实负载、容量收缩、放置(P3)

> 日期:2026-10-28。冻结门见 `docs/NIGHTLY-LOG.md` P3.1/P3.2。30B L2 一站式加载(514 s),TP2/EP2,grouped_mm(prefill)+ bmm_dense(decode),eager。

## 1. 真实负载统计(P3.1,fold 前)——首次真实 chat 负载测量

负载:32 条中英 chat prompt,512-token prefill + 64-token decode。

| 指标 | 值 | 解读 |
|---|---:|---|
| routing 总 slot | 6,868,992 | — |
| folded 无效 slot 占比 | **47.1%** | 与架构书 §2.2 推导 ~50% 吻合;每层一半 GEMM 行是零权重 |
| expert 负载 Gini | 0.296 | 层内倾斜显著 |
| **rank imbalance(逐层均值)** | **15.8%** | 首层 rank0 63,654 vs rank1 79,450;**远超 synthetic 的 1.7–5.2%** |
| S_max 观察(b32 decode) | 中位 25 / 最大 32 | 最坏 256;收缩机制按设计工作 |

**结论:①"负载均衡 5%"确实是被 synthetic 掩盖的假象,真实 chat 负载 imbalance 15.8%,profiled placement 有真实动机;②47% 折叠浪费是常态,不是偶发。**

## 2. bmm_dense 容量收缩 A/B(P3.2)——数值全过、性能否决

实现:owned-only 打包(dump-slot scatter,无布尔索引)+ `S = counts.max()` 动态容量 + 每层 1 次 stacked host sync(S 与 n_owned 同步);capture 时自动回退 padded 路径。

| 批 | padded 中位 | shrunk 中位 | delta | CV(p/sh) | token |
|---:|---:|---:|---:|---|---|
| b32 | 308.47 ms | 303.91 ms | **−1.5%** | 1.0%/0.2% | **全等(identical=True)** |
| b64 | 439.41 ms | 439.89 ms | **+0.1%** | 0.2%/0.1% | **全等** |

- 数值门:PASS(且优于预期——行级 K 维归约与 M 位置无关,bitwise 一致,无需 L2 软门)。
- 性能门(−10%):**FAIL**。bmm 行数 ~10× 削减不动步时,与 P1 的 fused RMSNorm 结论互证:**decode 步时不在 MoE bmm 工作量上**。
- **判定:不采用(默认关,`NANOVLLM_MOE_CAPACITY_SHRINK=1` 可开);设计、统计钩子(`set_shrink_stats`)与 A/B 脚本保留。**

## 3. placement A/B(P3.3)——未跑,原因与就绪度

- 未跑原因:换 placement 需重载 27 GiB 权重(≈2×7 min),时间与余额让位给验收 D/E;真实 imbalance 15.8% 是新动机。
- 就绪度:`scripts/profile_experts.py`(LPT 生成器)+ 本轮 fold 前 owned 统计可直接产出 `p8_expert_map.json`;A/B 命令与门(吞吐 +5% 或 imbalance −10%)已在计划 §4.4 冻结。**下轮第一优先。**

## 4. fused route/pack 微基准(P3.4)——未跑

route/pack 558.8 ms 的旧 profile 来自 grouped_mm 路径(b32/c2048);本轮 decode 主路径已是 bmm_dense,该数字的前提变化。按计划 §10/§11 顺延,不做猜测性结论。

## 5. 证据

`/root/p8_moe_30b.json`(stats/capacity_ab/prefill_sync_ab 三段)、`/root/p8_moe_30b.log`;sha256 见 `benchmarks/P8-nightly.md`。
