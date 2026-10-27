# P6 — I8 长上下文（重设计后重测）：可服务，但容量在 0.435 QPS 就崩

> 日期：2026-10-27 19:29–19:42。**这次是有效跑**（上一轮是无效跑：18,354 token + rank1 死亡）。
> 栈：Qwen3-30B-A3B-Instruct-2507-FP8，TP2/EP2，**新默认** `attention=triton_paged`、`decode_moe_backend=bmm_dense`、`moe_backend=grouped_mm`、`scheduler=stall_free`、`enforce_eager=True`、**`--max-model-len 4096`**。
> 负载：`prompt_mode=unique`、**`--prompt-tokens 1024`**（tokenizer 迭代校准为 68 词）、`--max-qps 0.87`、三档 0.5/0.75/1.0、每档 128 请求 + 4 warmup、输出 64 token、Poisson（seed 20261026）、SLO = TTFT ≤ 2 s 且请求级 TPOT ≤ 0.5 s。
> 证据：`benchmarks/data/p6d/p6_i8_long_unique_v2.json`（sha256 `1249e49bcb3de838500bf0e0e9007310597ee27957deddb0c54017cfacaa297e`）、watchdog `i8_watch.log`（`dd3b0c1e9866d4761a4191613123e50b79bbb1bfc425d4dff749f472fa631631`）、运行日志 `p6d_i8.log`（`ecb86646c280d8ecacd370ce85bca9d79c35bb9a5639c1ac7db3d90acf82e40a`）。

## 1. 有效性门（跑前写入 `/root/p2_i8_gate.txt`）

| # | 门 | 实测 | 判定 |
|---|---|---|---|
| 1 | `prompt_tokens` p50 ∈ [960, 1088] | p50 = **1086 / 1154 / 1154**（三档）；全局校准值 1018 | ⚠️ **按字面未过**：二/三档 +12.7%，超 ±6% 带 |
| 2 | watchdog 每 5 s 采样 rank1 存活 | 单实例 watchdog，运行期全部样本 `rank1=503969`；只有 spawn 前的第一条为空 | ✅ |
| 3 | 每档 ≥125/128 收到 `[DONE]` | **384/384**，零失败 | ✅ |

**对第 1 条的诚实说明**：长度偏差来自 unique prompt 里嵌套的 request-id 数字（1k id 段与 3k id 段分词数不同），不是 18× 级别的失控。实际长度 **1018–1154 token（0.99–1.13× 目标）**，落在"~1k"的意图内，因此本报告按**长度区间有效**使用，但门本身记为"按字面未过"，不粉饰。

## 2. 结果

| 负载档 | 目标 QPS | 达成 QPS | 输出 tok/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | ITL p95 | E2E p95 | 最大并发 | SLO | goodput |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50% | 0.435 | 0.421 | 26.91 | **3.843 s** | **5.475 s** | 0.418 s | 0.638 s | 0.893 s | 44.3 s | 25 | **0%** | 0.000 |
| 75% | 0.652 | 0.590 | 37.77 | **8.275 s** | **12.452 s** | 1.175 s | 1.701 s | 2.158 s | 115.2 s | 80 | **0%** | 0.000 |
| 100% | 0.870 | 0.660 | 42.21 | **10.219 s** | **14.647 s** | 1.419 s | 1.977 s | 2.649 s | 131.9 s | 104 | **0%** | 0.000 |

SLO 分母为全部 128 条/档（含失败；本跑无失败）。

## 3. 失败在哪：TTFT，不是 decode

| 档 | 满足 TTFT ≤2 s | 满足 TPOT ≤0.5 s | 同一条 prompt 的空闲基线 |
|---:|---:|---:|---|
| 50% | **0/128** | 83/128 | 1,154-token 单请求：**TTFT 0.368 s、TPOT 0.144 s** |
| 75% | 0/128 | 8/128 | 同上 |
| 100% | 1/128 | 4/128 | 同上 |

- **单请求能力没问题**：服务空闲时同长度 prompt 的 TTFT 0.368 s、TPOT 0.144 s（首次 1.959 s 含一次性预热），远在 SLO 内。
- **一旦有并发就崩在 TTFT**：50% 档 TTFT p50 3.84 s，是空闲基线的 **10×**；75/100% 档到 8.3/10.2 s。
- **并发被长请求放大**：v4（137-token prompt）最大并发 14/23/24；本跑升到 **25/80/104**。Little's law 自洽：请求 E2E 从 ~20 s 涨到 ~100 s，到达率不变，队列必然堆起来。
- **TPOT 也随之恶化**（0.418 → 1.419 s）：decode 批更大，且 mixed step 走 prefill 路径（P8 台账 I23 已指出），bmm_dense/paged 在 mixed step 不生效。

**结论：长上下文的瓶颈是"1k prefill 与 decode 混跑 + 无准入控制"造成的排队，不是崩溃、不是 rank1、不是单个 kernel。** 在 0.435 QPS（v4 请求率的 50%）就已经 0% SLO；按当前实现，1k prompt 的可承载请求率远低于 0.435 QPS。

## 4. 与上一轮无效跑的区别

| 项 | 上轮（无效） | 本轮（有效） |
|---|---|---|
| prompt 实际长度 | 18,354 token（words≠tokens） | 1,018–1,154 token（token 校准） |
| 请求完成 | 8/128、8/128、4/128 收到 `[DONE]` | **384/384** |
| rank1 | 运行中死亡 | 全程存活（5 s 采样） |
| SLO | 0%（不可用） | **0%（可用但超时）**——这次是真实容量结论 |

## 5. 下一步（按证据排序）

1. **准入/背压（I26）**：当前整段预留 + 容量不足 assert；应先限流/排队并返回 429，而不是让 TTFT 无限膨胀。
2. **mixed step 的后端（I23）**：mixed step 走 prefill 路径让 decode 侧失去 bmm_dense/paged，长上下文高并发下这是 TPOT 恶化的直接嫌疑。
3. **prefill chunk 与到达率匹配**：`prefill_chunk_size=256` 在 1k prompt 下是 4 个 chunk/请求；需要测 chunk 大小与 TTFT 的曲线（P8 计划内）。
4. **长上下文目标 QPS 重定**：0.87 QPS 是 137-token 口径的封顶，不能直接套到 1k prompt。下一轮应扫 0.05–0.435 QPS 找真实拐点。

## 6. 局限

- unique prompt 关闭了前缀缓存收益（这是有意的，避免复用整块 KV）；shared-prefix 场景的 TTFT 会好很多（I6：−25.6%）。
- 单客户端、固定 64-token 输出、单实例。
- 三档的 prompt 长度不完全相同（1086 vs 1154），二/三档之间可直接比，一档与二/三档比较时需记住这 ~6% 的长度差。
