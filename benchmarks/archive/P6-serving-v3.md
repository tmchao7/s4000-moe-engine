# P6 — 在线容量 v3（sdpa + decode bmm_dense，Poisson 开放环）

> 日期：2026-10-27 00:47–01:02（第 3 发 30B 加载）。
> 栈：Qwen3-30B-A3B-Instruct-2507-FP8（TP2/EP2），`attention=sdpa`、`moe_backend=grouped_mm`、**`decode_moe_backend=bmm_dense`**、`scheduler=stall_free`、`enforce_eager=True`。
> 与 `benchmarks/P5-serving-v2.md` 唯一变量 = decode MoE 后端（v2 是 grouped_mm；P0 门通过后采用，见 `docs/NIGHTLY-LOG.md`）。
> 负载/SLO/客户端参数与 v2 完全一致：Poisson 开放环（seed 20261026），`--max-qps 0.87`，四档 0.5/0.75/0.9/1.0，每档 128 请求 + 4 warmup，输出 64 token（`ignore_eos` 关），prompt 实测 137 token；SLO = TTFT ≤ 2 s 且请求级 TPOT ≤ 0.5 s。
> 证据：`benchmarks/data/online-v3.json`（原始，含 512 条请求记录）。

## 1. 结果（同表对比 v2）

| 负载档 | 目标 QPS | 达成 QPS | 输出 tok/s | TTFT p95 | TPOT p50 | TPOT p95 | ITL p95 | E2E p95 | 最大并发 | SLO 达标率 | goodput (req/s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50% | 0.435 | 0.427 | 27.31 | 0.816 s | 0.199 s | **0.229 s** | 0.460 s | 14.9 s | 14 | **100.0%** (128/128) | **0.427** |
| 75% | 0.652 | 0.656 | 42.01 | 1.073 s | 0.297 s | **0.348 s** | 0.682 s | 22.8 s | 24 | **100.0%** (128/128) | **0.656** |
| 90% | 0.783 | 0.706 | 45.18 | 1.126 s | 0.268 s | **0.312 s** | 0.596 s | 20.4 s | 21 | **100.0%** (128/128) | **0.706** |
| 100% | 0.870 | 0.771 | 49.35 | 1.270 s | 0.318 s | **0.401 s** | 0.741 s | 26.3 s | 27 | **100.0%** (128/128) | **0.771** |

（SLO 达标率由 512 条请求记录逐条重算：`ttft_s ≤ 2.0 and tpot_s ≤ 0.5`。四档全部 128/128 成功、全部收到 SSE [DONE]。）

## 2. 与 v2 同表对比（decode MoE 是唯一变量）

| 负载档 | v3 SLO% (TPOT p95) | v2 SLO% (TPOT p95) | v3 goodput | v2 goodput | goodput 倍数 |
|---:|---:|---:|---:|---:|---:|
| 50% | **100.0%** (0.229 s) | 100.0% (0.455 s) | 0.427 | 0.411 | 1.04× |
| 75% | **100.0%** (0.348 s) | 26.6% (0.697 s) | 0.656 | 0.163 | 4.02× |
| 90% | **100.0%** (0.312 s) | 11.7% (0.685 s) | 0.706 | 0.077 | 9.17× |
| 100% | **100.0%** (0.401 s) | 7.0% (0.694 s) | 0.771 | 0.050 | 15.42× |

- **100% SLO 的负载档：v2 的 50% → v3 的全部四档（含 100%）**。夜间目标（推到 ≥75%）超额达成。
- TTFT 全程不再是约束：v2 在 100% 档 TTFT p95 已到 2.067 s（越线）；v3 全档 ≤ 1.27 s。
- 输出吞吐 49.35 tok/s @100%（v2 45.72），但这是 **SLO 内**的吞吐：v2 的 45.7 tok/s 是靠 93% 请求违反 TPOT SLO 换来的。

## 3. 怎么读

1. **瓶颈迁移**：v2 的拐点被 decode 单步成本卡住（TPOT 随并发恶化）；v3 把 b32 decode 的 MoE 部分砍掉一半以上（离线 TPOT p95 @b32 0.335→0.249 s），在线 TPOT p95 全档 0.23–0.40 s，四档齐过。SLO 拐点被推到 >100% 负载（即当前 0.87 QPS 封顶内不存在拐点）。
2. **90% 档 TPOT（0.312）低于 75% 档（0.348）**：非单调，来自 stall_free 调度在两档的并发/混步差异（最大并发 21 vs 24）。不影响结论。
3. **与离线矩阵交叉**：离线 sdpa+bmm_dense b32 = 110.6 tok/s（`docs/NIGHTLY-LOG.md` P0 表）；在线 100% 档 49.35 tok/s @并发 27。差异仍来自在线 64-token 输出 + 混步 + 到达间隔，口径不同，不能直接比。
4. **下一步的容量空间**：四档全绿后，容量提升只剩 (a) 更高注入率（>0.87 QPS，本轮未测——v2 口径上限就是 0.87）、(b) triton_paged（离线 TPOT 再 -20~31%，但一致性门未过，见 P1）、(c) decode 图（replay 未解）。

## 5. 复现命令（第 3 发加载）

```bash
cd /root/moe-server-s4000 && export PYTHONPATH=/root/moe-server-s4000
MODEL=/data/models/Qwen3-30B-A3B-Instruct-2507-FP8
python3 -m nanovllm.entrypoints.openai.api_server --model "$MODEL" \
  --served-model-name qwen3-30b-a3b --host 127.0.0.1 --port 8000 \
  --scheduler-policy stall_free --decode-moe-backend bmm_dense > /root/p6_server.log 2>&1 &
# 等 /health 后：
python3 scripts/benchmark_online.py --model qwen3-30b-a3b --tokenizer "$MODEL" \
  --max-qps 0.87 --requests 128 --max-tokens 64 --prompt-words 128 \
  --arrival-pattern poisson --slo-ttft-ms 2000 --slo-tpot-ms 500 \
  --fractions 0.5,0.75,0.9,1.0 --output benchmarks/data/online-v3.json
```

## 6. 局限（沿 v2 §6 全部适用）

单实例/单客户端/固定形状（137-in/64-out）；SLO 按请求级 TPOT 而非逐 token ITL；`achieved_qps` 分母含尾部长尾；0.87 QPS 是 v2 口径的封顶——「100% 档全绿」不等于「0.87 QPS 就是饱和点」，只说明在 v2 口径内 SLO 不再是约束。
