# P5 — 在线容量 v2（Poisson 开放环 + goodput）

> 第 5 发加载（model_load_index=5）。日期 2026-10-26 19:03–19:19。
> 证据：`benchmarks/data/online-v2.json`（原始 1.2 MB，含 512 条请求记录）、`online-v2.jsonl`、`online-v2.md`。
> 栈：Qwen3-30B-A3B-Instruct-2507-FP8（加载期反量化 BF16），TP2 + EP2，`attention=sdpa`、`moe=grouped_mm`、`enforce_eager=True`、`scheduler=stall_free`、`kvcache_block_size=256`。
> SLO：TTFT ≤ 2000 ms 且请求级 TPOT ≤ 500 ms（两者同时满足才算 good）。

## 1. 负载定义（可复现）

- 客户端 `scripts/benchmark_online.py`，**Poisson 开放环**（`--arrival-pattern poisson`，seed 20261026），每档 128 请求 + 4 条不计入统计的 warmup。
- 输入：同一段词重复 128 次 → tokenizer 实测 **137 prompt tokens**；输出上限 **64 tokens**，`ignore_eos` 关闭（实测 512/512 请求都 `finish_reason=length`、`output_tokens=64`）。
- 目标注入率 `--max-qps 0.87`，四档 `--fractions 0.5,0.75,0.9,1.0`。
- 正确性硬校验：每个请求必须收到 SSE `[DONE]`，token 数用服务端 usage 计。**512/512 通过**（`done_received=true`），成功率 100%。

## 2. 结果

| 负载档 | 目标 QPS | 达成 QPS | 输出 tok/s | 总 tok/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | ITL p95 | E2E p95 | 最大并发 | SLO 达标率 | goodput (req/s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50% | 0.435 | 0.411 | 26.31 | 82.62 | 0.830 s | 1.137 s | 0.404 s | **0.455 s** | 0.679 s | 29.58 s | 19 | **100.0%** (128/128) | **0.411** |
| 75% | 0.652 | 0.615 | 39.38 | 123.68 | 1.164 s | 1.784 s | 0.558 s | 0.697 s | 1.065 s | 45.22 s | 42 | 26.6% (34/128) | 0.163 |
| 90% | 0.783 | 0.658 | 42.13 | 132.31 | 1.352 s | 1.932 s | 0.619 s | 0.685 s | 1.104 s | 44.80 s | 39 | 11.7% (15/128) | 0.077 |
| 100% | 0.870 | 0.714 | 45.72 | 143.59 | 1.308 s | 2.067 s | 0.616 s | 0.694 s | 1.126 s | 44.99 s | 45 | 7.0% (9/128) | 0.050 |

（`SLO 达标率` 由 512 条请求记录逐条重算：`ttft_s ≤ 2.0 and tpot_s ≤ 0.5`，与脚本的 `goodput` 定义一致。）

## 3. 怎么读这张表

1. **SLO 拐点落在 50%→75% 之间，而且是被 TPOT 卡住的**：50% 档 TPOT p95 0.455 s 还在 0.5 s 以内；75% 档 TPOT p50 已 0.558 s，直接越线。TTFT 全程 < 2.1 s，**不是 TTFT 先崩**。
2. **吞吐与 goodput 反向**：输出吞吐从 26.3 → 45.7 tok/s（+74%），但 SLO goodput 从 0.411 → 0.050 req/s（-88%）。也就是说这 74% 的吞吐是用违反 SLO 换来的，**对"服务"没有价值**。这是"容量"与"吞吐"必须分开报的直接证据。
3. **并发没有换来线性吞吐**：最大并发 19 → 45（+2.4×），输出吞吐只 +1.74×，TPOT 从 0.40 → 0.62 s。decode 每步成本随 batch 上升而上升（prefill 混入 + KV gather/SDPA 变长），所以"堆并发"在当前栈上很快失效。
4. **达成 QPS < 目标 QPS 是口径效应，不是丢包**：`achieved_qps = 成功请求数 / (最后一个请求结束 − 第一个请求开始)`，分母含尾部长尾（E2E p95 有 45 s），因此高负载档被低估。同时 `arrival_lag`（计划发送 vs 实际发送）p95 仅 **2 ms**，说明客户端确实准点按 Poisson 注入，开放环成立。
5. **与离线矩阵交叉对照**：离线 b16（prompt 128 / out 32）= 40.57 tok/s，b32 = 69.39 tok/s；在线 100% 档在并发 45 时只有 45.7 tok/s。差异来自三点：在线每请求要输出 64 token（离线 32）、prefill 与 decode 混步（100% 档 `mixed_steps=97`、`prefill_steps=1`）、以及 SLO 约束下调度无法随意堆 batch。**在线数字不能拿来和离线批测直接比大小**，只能各自比自己。

## 4. 这条曲线说明什么、下一步该做什么

- **当前瓶颈是 decode 的单步成本**：`docs/P5-profiling.md` 已经量到长上下文下 `gather+mask` 占 forward 31%–58%、MoE 占 27%–53%；本期在线数据在"短上下文（137 tokens）"下同样表现为 TPOT 随 batch 单调变差。想抬高 SLO 容量，只能降低单步成本，而不是加并发。
- 因此下一步的顺序是 **(a) attention 换 Triton paged**（同后端 +19%~+59%，但被 b32 可复现性挡住，见 `benchmarks/P5-attention-ab.md`）、**(b) decode 图捕获**（`docs/P6-graph-capture.md`，当前在重审）、(c) MoE autotune。
- **SLO 门槛本身要写清楚**：本仓库默认 SLO 是 TTFT 2 s / TPOT 500 ms。若按 TPOT 250 ms 报，连 50% 档都不达标；若放宽到 800 ms，100% 档也全过。**换 SLO 就是换容量结论**，任何对外数字必须带 SLO 一起写。

## 5. 复现命令（第 5 发加载）

```bash
# 服务器端（/root/p5_online_run.sh 的核心，一次跑完）
cd /root/moe-server-s4000 && export PYTHONPATH=/root/moe-server-s4000
MODEL=/data/models/Qwen3-30B-A3B-Instruct-2507-FP8
python3 -m nanovllm.entrypoints.openai.api_server --model "$MODEL" \
  --served-model-name qwen3-30b-a3b --host 127.0.0.1 --port 8000 \
  --scheduler-policy stall_free > /root/p5_server.log 2>&1 &
# 等 /health 返回 {"status":"ok"} 后：
python3 scripts/benchmark_online.py --model qwen3-30b-a3b --tokenizer "$MODEL" \
  --max-qps 0.87 --requests 128 --max-tokens 64 --prompt-words 128 \
  --arrival-pattern poisson --slo-ttft-ms 2000 --slo-tpot-ms 500 \
  --fractions 0.5,0.75,0.9,1.0 --output benchmarks/data/online-v2.json
```

## 6. 局限（写清楚，避免被当成工业容量结论）

1. 单实例、单模型、单客户端；无负载均衡、无多副本、无真实 trace。
2. 输出固定 64 token、输入固定 137 token：只覆盖"短输入中输出"一种形状。长输入（≥1k）的 prefill 主导区间未测。
3. SLO 达标率是按"单请求是否达标"算的，不是按请求里每个 token 的 ITL 算；两者在高负载档会差。
4. `achieved_qps` 分母含尾巴，见 §3.4；跨档比较请用 `输出 tok/s` 和 `SLO 达标率`。
