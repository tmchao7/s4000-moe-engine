# P9 — 服务化补全:Poisson ×3 重复与 Wilson CI(P4)

> 日期:2026-10-28 13:52–14:41。门冻结于 `docs/PLAN.md` 回归门(vs v4)与 P9 计划 P4。
> schema_version 3;slo_definition 双口径(standard: TTFT≤2s 且请求级 TPOT≤0.5s;strict: + ITL≤1s);prompt_mode = prompt-words 128(p50=137 token,v4/v5 口径);warmup_done yes(4 warmup/点);percentile_method linear;p99 仅参考。
> 证据:`benchmarks/data/p9/p9_serving_v5_ci.json`、`p9_l4_client.log`、`p9_l4_server.log`(sha256 见 P9-nightly)。

## 1. 设置

服务器 = 默认生产栈(triton_paged / grouped_mm prefill / bmm_dense decode / eager / stall_free / chunk-256,与 v4 同口径);客户端:`--mode poisson --max-qps 0.87 --fractions 0.5,0.75,0.9,1.0 --requests 128 --prompt-words 128 --repeats 3 --request-timeout-s 120`。1,536/1,536 请求成功,错误桶全 none,零 429/timeout。

## 2. 逐点结果(12 点 = 4 档 ×3 repeats)

| 档 | SLO std(per-repeat) | SLO strict(per-repeat) | achieved QPS 中位 | Wilson 95% CI(std,合并) |
|---|---|---|---|---|
| 0.5 | 100%,100%,100% | 100%,100%,100% | 0.408 | [99.0%, 100%] |
| 0.75 | 100%,100%,100% | 100%,100%,100% | 0.611 | [99.0%, 100%] |
| 0.9 | 100%,100%,100% | 100%,100%,100% | 0.665 | [99.0%, 100%] |
| 1.0 | 100%,100%,100% | 100%,100%,100% | 0.713 | [99.0%, 100%] |

延迟(p50/p95,s 范围跨 repeats):

| 档 | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 |
|---|---|---|---|---|
| 0.5 | 0.749–0.770 | 1.055–1.120 | 0.309–0.318 | 0.332–0.350 |
| 0.75 | 0.775–0.802 | 1.066–1.127 | 0.349–0.354 | 0.375–0.388 |
| 0.9 | 0.801–0.810 | 1.157–1.173 | 0.362–0.365 | 0.381–0.382 |
| 1.0 | 0.783–0.798 | 1.192–1.197 | 0.372–0.376 | 0.387–0.397 |

run-to-run 极差:TPOT p95 ≤0.018 s、TTFT p95 ≤0.065 s(全部 ≤6% 相对极差)——重复非常稳定。

## 3. 容量判决

**slo_standard ≥99% 的最大达成 QPS = 0.713(= 1.0 档,即本次测试的最高档;四档全过,真实饱和点未 bracketed,在 0.87 target 之上)**,合并 Wilson 95% CI [99.0%, 100%],run-to-run 极差 0。双 SLO 并列:strict 亦全部 100%。

## 4. 回归门(vs v4,0.87 档)

| 门 | 阈值 | 实测 | 判定 |
|---|---|---|---|
| SLO standard | 100% | 100% ×3(CI [99.0%,100%]) | **PASS** |
| TTFT p95 | ≤1.263 s | 1.192–1.197 s | **PASS** |
| TPOT p95 | ≤0.343 s | **0.387–0.397 s(+13~16%)** | **FAIL** |

**判定:部分回归。** TPOT p95 超门且跨 repeats 稳定(非噪声)。候选解释(未本轮判定,转 P10):① **I23 mixed-split 的短 prompt 税**——137-token 负载混步频繁,拆分后 decode 行走独立 paged+bmm_dense 前向,相对旧"整批走 prefill 路径"多一次前向切换;② P8 期间 decode 路径改动的累积效应;③ v4 与本轮的服务端背景差异。建议 P10 做 `NANOVLLM_MIXED_SPLIT=0/1` 在线 A/B(同矩阵)。注意 SLO 门(决策相关门)与 TTFT 门均 PASS,容量结论不受影响。

## 5. 未做

1.0 档以上的饱和探索(fractions >1.0)未做——四档全 100% 使"最大 SLO 达标 QPS"未 bracketed;P8 v5 并发模式的饱和点(~2.14 req/s)可作为参照,P10 可补 Poisson 1.2/1.4 档。
