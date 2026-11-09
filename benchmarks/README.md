# Benchmarks

> **开发/审计证据，不是产品部署材料。** 发布包不包含本目录；部署只需要 `docs/DEPLOY.md`。

实验报告目录。**当前轮报告在根目录**(P9-*),**P0–P8 历史报告在 `archive/`**;原始 JSON/JSONL 证据在 `data/`(gitignore,本地保留,sha256 写在各轮 nightly 总表里)。冻结门的逐实验日志见 git 历史 `docs/NIGHTLY-LOG.md`。

## 当前轮(P9)

- `P9-nightly.md` — 总表 + 全部证据 sha256 索引(入口)
- `P9-graph.md` — I29 图楔死分级探针与判决(0.6B TP2 最小复现)
- `P9-moe.md` — profiled expert placement A/B(imbalance −99.5% 但吞吐 +0.01%)
- `P9-long-context.md` — chunk 256 vs 2048 对照(1k TTFT −69~73%)
- `P9-serving.md` — Poisson ×3 四档双 SLO + Wilson CI + TPOT p95 回归(I30)

## 历史(`archive/`,P0–P8,26 份)

每轮以 `P*-nightly.md` 为入口(总表 + sha256);关键判决:attention 五层等价门(`P6-L3-review.md`、`P6-attention-equivalence-decision.md`)、bmm_dense 采用(`P6-nightly.md`)、投机解码 NO-GO(`P7-phase0.md`)、I19/图/MoE/长上下文(P8-* 总表 `archive/P8-nightly.md`)。

冻结决策 JSON(阈值不可改):`p5-profile-decision.json`、`p5-attention-ab-decision.json`、`p5-moe-ab-decision.json`;门的实现是 `scripts/p5_decide.py` / `scripts/p6_decide_attention_equivalence.py`。

## 统一测量方法

- **黄金规则**:任何数字只有在正确性门通过后才有意义;旧 24.4 tok/s(乱码输出)已作废。
- **正确性**:`temperature=0` 贪心;tiny 模型六配置(TP1/TP2 × loop/grouped/bmm_dense)逐 token 全等;跨后端比较用五层等价门(`docs/numerics-policy.md`)。
- **离线吞吐**:每点 warmup 1 次、正式 ≥3 次;CV = sample std / mean;A/B 交错 ≥5 次、中位数、CV ≤5% 否则 INVALID。
- **在线压测**:Poisson 开放环;收到 `[DONE]` + 真实 usage 才算成功;容量结论 ≥128 请求/点 ×3 repeats,Wilson 95% CI;双 SLO(standard/strict)并列,p99 仅参考。
- **已知偏差**:贪心采样(temperature=0),与生产 temperature=0.6 口径不同;对比外部数字时须标注。
