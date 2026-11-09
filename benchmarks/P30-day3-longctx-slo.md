# P30-A Day3：8k 并发/SLO 探针（30B）

> 日期：2026-11-09。范围：重跑 Day2 被中止的 `scripts/p30_longctx_slo.py`，补上 8k 并发 SLO 数字，给 8k 能力定位（交互 vs 离线/批处理）下判决。
> 原始 JSON：`benchmarks/data/P30/longctx_slo.json`（`benchmarks/data/` 不入库）。
> 脚本：`scripts/p30_longctx_slo.py`（本次未改动，按 Day2 写好的口径原样跑）。

## 1. 预注册口径（继承 Day2，未改）

- 18 条独立 7600-token **随机 token** prompt（无前缀复用，worst-case prefill/排队）；
- `max_model_len=8192`、`max_num_batched_tokens=16384`、`gpu_memory_utilization=0.85`（opt-in 配置，非生产默认 4096）；
- 生产栈：TP2/EP2、`grouped_mm` prefill、`bmm_dense` decode、`triton_paged`、eager、stall_free、chunk 256；
- greedy `max_tokens=4` + `ignore_eos`（测量 prefill/排队为主，decode 仅收尾）。

## 2. 结果（rc=0，一次加载窗口内完成）

| 指标 | 值 |
|---|---:|
| requests × prompt_len | 18 × 7600（合计 136,800 prompt tokens） |
| output tokens | 72（18×4，全部完成，无抢占/丢弃） |
| wall | 78.999 s |
| TTFT p50 / p95 / max | **63.52 / 63.52 / 78.31 s** |
| e2e p50 / p95 / max | 65.375 / 65.375 / 78.997 s |
| 聚合输出吞吐 | 0.91 tok/s（输出太少，此数无意义，仅记录） |

p50==p95 是**真并列**：18 条里 16 条在同一步拿到首 token（chunk-256 轮转 + 16384 batched tokens/步 ⇒ 每步 ~64 个 chunk，全部 18 条在 ~9 个大步内交织 prefill 完，多数 seq 的最后一块落在同一步）。`percentile` nearest-rank 实现已核对，非分位数 bug。

### 派生数字（口径注明）

- **并发 prefill 聚合吞吐 ≈ 2.1k tok/s**（136,800 / ~64s，TTFT max 前完成主体）；
- 对照单流 Day8：7.6k prefill ~14 s ⇒ ~543 tok/s；18 并发把聚合吞吐放大 **~3.9×**（launch-bound eager prefill 被并发摊销）；
- 引擎 init（冷加载）：8.6 min（rank0 weights_loaded dt=501 s 为大头；warmup ~25 s；kv_allocated 0.06 s）。

## 3. 判决：8k = 离线/批处理能力，非交互能力

- 交互 SLO 门是 TTFT ≤ 2 s（v3 口径）。本探针 **TTFT p50 63.5 s，超门 ~32×**；单流 8k prefill 也要 ~14 s，同样不可交互。
- 结论与 P30-OPPORTUNITIES §A 的预估一致（8k prefill ~14s ⇒ "先决定场景"）：**8k 定位为 opt-in 的离线/批处理能力**——长文档摘要、批量知识抽取、离线 needle/QA 评测。
- 并发端的好消息：KV 池（util 0.85）装得下 18 条 7.6k 并发，全员完成、无抢占无 429；并发把聚合 prefill 吞吐抬到 2.1k tok/s，批处理场景有效。

## 4. 产品化收口

1. 生产默认 `max_model_len=4096` **不变**；
2. 8k 用法写入 [OPERATIONS.md](../docs/OPERATIONS.md)（opt-in `--max-model-len 8192` + 预期时延口径：单流 prefill ~14 s、18 并发 TTFT p50 ~64 s，只承诺离线/批处理）；
3. P30-A 验收四项至此全齐：质量门 9/9（[day2](./P30-day2-longctx-quality.md)）、API 超限 400（day2 §3）、并发压测（本文）、opt-in 文档（OPERATIONS.md）。

## 5. 边界

- 单环境单负载：随机 token 无前缀复用，真实长文档负载（前缀共享）TTFT 会好于本数；
- `--repeats` 未做：18×7.6k 单发已足够下"离线定位"判决，吞吐倍率 3.9× 是单发口径，不作为承诺值引用；
- max_tokens=4：decode 侧 8k 并发 TPOT 未测（Day8 b1 210.8 ms 是单流口径），批处理吞吐评估应以 prefill 为主。
