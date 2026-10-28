# P9 — profiled expert placement A/B(P2)

> 日期:2026-10-28 12:46–13:06(两次 30B 加载)。门冻结于 git 历史 `docs/NIGHTLY-LOG.md` P9 段。
> schema_version 1;prompt_mode = P8 冻结 chat 负载(32 条中英 ×512 token,unique filler);warmup_done yes;percentile_method median(墙钟中位数)。
> 证据:`benchmarks/data/p9/p9_placement_linear.json`、`p9_placement_profiled.json`、`p9_expert_map.json`(sha256 见 P9-nightly)。

## 1. 方法

placement 是加载期配置 → 跨加载 A/B,每臂一次 30B 加载,同一冻结工作负载(`tests/p9_placement_ab.py`):

- decode b32 ×3(32 chat ×512 token prompt,decode 64)→ 每步墙钟中位数 + CV;
- prefill-512 b32 ×3 → prefill 墙钟中位数;
- 排名失衡 = A/B 负载的 global routing counts × 各臂**实际 ownership**(linear = 连续 64/64;profiled = linear 臂同负载 counts 的 per-layer balanced partition map,带 model config sha256)。

## 2. 结果

| 指标 | linear | profiled | Δ | 门 |
|---|---|---|---|---|
| decode b32 中位墙钟 | 303.28 ms(CV 0.0003) | 303.30 ms(CV 0.0085) | **+0.01%** | ≤0.95×(吞吐 +5%)**FAIL** |
| prefill-512 中位墙钟 | 2266.4 ms | 2257.1 ms | −0.4% | 报告项 |
| rank imbalance(均值) | 0.1685(profiled 臂复测 0.1687,跨臂一致) | **0.0009** | **−99.5%** | ≤0.90×(−10%)**PASS** |

## 3. 判决

**imbalance 门大幅通过、吞吐门完全不动 → 动机假设("rank 失衡是 30B decode 性能瓶颈")被否定。决策:不采用为默认**;`--expert-placement profiled` 与 map 生成流程保留为已验证工具。

科学含义:15.8%(真实 chat 负载)的 rank 失衡对 b32 decode 墙钟无可测影响——与 P8 I19 判决(步时由 ~20–30 ms 未解释固定等待主导、对 device 工作量不敏感)互证;P8 容量收缩(−10× MoE 行数)步时不动是同一现象的第三个证据。若未来负载呈 rank-bound 特征(imbalance >25% 且 device-bound),本流程可直接复用。

## 4. 备注

- profiled map 生成:`tests/p9_placement_ab.py --arm linear --make-map`(同负载 counts → per-layer greedy balanced partition);map 自带 model config sha256,加载时校验(`qwen3_moe._expert_ids`)。
- 跨臂 token 流:路由与 ownership 无关(路由在权重上),greedy 下两臂 token 应一致;未设门,记录于 JSON。
