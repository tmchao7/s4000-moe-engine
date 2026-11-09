# P29 Day4：linear vs round-robin 重复 A/B + 30B token 门

> 日期：2026-11-09。范围：P29-E 的重复 A/B 与 30B 贪心 token 门；不改生产默认。
> 原始 JSON：`benchmarks/data/P29/rr_ab_linear.json`、`benchmarks/data/P29/rr_ab_rr.json`（`benchmarks/data/` 不入库）。
> 脚本：`scripts/p29_rr_ab.py`。

## 1. 实验设计

- 每个 placement 单独加载一次 30B（placement 在 load 时固定），生产栈：TP2/EP2、`grouped_mm` prefill、`bmm_dense` decode、`triton_paged`、`stall_free`、chunk 256、eager。
- token 门：`benchmarks/p2-reference-8tok.json` 的 4 条固定 prompt，贪心 8 token，与参考 token 全等才算 PASS。
- 性能：`1x128, 8x128, 32x128`，每档重复 3 次，每次 prefill 后测 8 个 clean decode step，取中位数；无 profiler。
- 两个 placement 串行跑，避免并发 GPU 任务。

## 2. Token 门：linear PASS，round_robin FAIL

| placement | token gate | 失败位置 |
|---|---|---|
| linear | **PASS** | — |
| round_robin | **FAIL** | 第 4 条 prompt（Python 快速排序）从第 1 个 token 起就不同 |

第 4 条 prompt：

- expected / linear：`[103942, 73670, 6313, 100431, 101909, 106646, 100700, 25074]`
- round_robin：`[1406, 73594, 12669, 198, 750, 3974, 6860, 10939]`

前 3 条 prompt 两种 placement 完全一致；只有第 4 条从第一个 token 分叉。

## 3. 性能：RR 没有可用的速度收益

| 负载 | linear median step | round_robin median step | 差异 |
|---|---:|---:|---:|
| 1×128 | 174.82 ms | 174.49 ms | −0.2% |
| 8×128 | 190.52 ms | 189.53 ms | −0.5% |
| 32×128 | 241.38 ms | 237.79 ms | −1.5% |

差异都在跨加载噪声范围内；**没有看到 RR 的性能收益**。

## 4. 结论：RR 不能转正

- 虽然 Day3 看到 RR 把 imbalance 从 8.5% 压到 0.3–0.5%，但 **30B token 门 FAIL**，按仓库的“先判定再改默认”原则，不能作为默认或 opt-in 产品特性。
- 失败原因目前**未定**，两种可能：
  1. **数值敏感性**：expert 分配到不同 rank 改变了 all-reduce 的两路 partial sum，浮点加法顺序变化；第 4 条 prompt 可能本身是 near-tie，贪心首 token 翻转。
  2. **RR 权重装载/映射 bug**：`load_expert_weight` 理论上按 global id 落位，但 30B 上尚未做逐专家权重审计。
- 小型 `tests/p2_grouped_mm.py` 的 `tp2-grouped_mm-round_robin` 六配置全等（p2 rc=0），说明 tiny 上映射/数值都过；因此更像 30B 的 near-tie 被跨 rank 顺序放大，而不是简单 loader 错位。**但这只是假设，未证明。**
- **决定：RR 保持 NO-GO；生产默认仍为 linear；不再把 RR 当性能候选。**

## 5. 下一步（Day5）

1. 如果还要碰 placement：先做 **逐专家权重审计** + **首 token logits/top-2 margin**，把“数值敏感性”与“装载 bug”分开；否则不继续。
2. 更稳妥的下一方向是 **DP=2 纸面/单卡原型**：两个独立 server 副本不改变单副本数值，token 门天然可过；但需要先确认 FP8/W8 完整模型单卡内存与吞吐。
3. 量化速度线仍暂停；MATE 只在独立环境验证。