# P6 — 在线容量 v4（triton_paged 转正，Poisson 开放环）

> 日期：2026-10-27 18:49–19:04（30B 第 4 发加载）。
> 栈：Qwen3-30B-A3B-Instruct-2507-FP8（TP2/EP2），**`attention=triton_paged`（本报告后成为默认）**、`moe_backend=grouped_mm`（prefill）、`decode_moe_backend=bmm_dense`、`scheduler=stall_free`、`enforce_eager=True`。
> 与 `benchmarks/P6-serving-v3.md` 唯一变量 = attention 后端（v3 是 sdpa）。
> 负载/SLO/客户端参数与 v3 完全一致：Poisson 开放环（seed 20261026），`--max-qps 0.87`，四档 0.5/0.75/0.9/1.0，每档 128 请求 + 4 warmup，输出 64 token（`ignore_eos` 关），prompt 实测 137 token；SLO = TTFT ≤ 2 s 且请求级 TPOT ≤ 0.5 s。
> 证据：`/root/online-v4-paged.json`（sha256 `8c113f7ba3038781cecf3bbc63c14518facfec35622c354da3eb1129da446ff4`）、`/root/p6d_l5.log`（`2739d7b496bc92123acb92c6038a8e42353f2d5ce048684d560d79758e6794a0`）、服务端日志 `/root/p6d_paged_server.log`（`ac2b633c893a8895ccf9a465a7b2ab921e3083e7c6672c81d9abfa2da8e33fdc`）。

## 1. 结果

| 负载档 | 目标 QPS | 达成 QPS | 输出 tok/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | ITL p95 | E2E p95 | 最大并发 | SLO 达标率 | goodput (req/s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50% | 0.435 | 0.428 | 27.39 | 0.526 s | 0.780 s | 0.177 s | **0.201 s** | 0.440 s | 13.3 s | 14 | **100.0%** (128/128) | 0.428 |
| 75% | 0.652 | 0.663 | 42.41 | 0.606 s | 1.060 s | 0.198 s | **0.290 s** | 0.555 s | 19.2 s | 23 | **100.0%** (128/128) | 0.663 |
| 90% | 0.783 | 0.711 | 45.50 | 0.654 s | 1.043 s | 0.214 s | **0.257 s** | 0.565 s | 16.8 s | 18 | **100.0%** (128/128) | 0.711 |
| 100% | 0.870 | 0.776 | 49.64 | 0.712 s | 1.203 s | 0.246 s | **0.327 s** | 0.688 s | 21.3 s | 24 | **100.0%** (128/128) | **0.776** |

512/512 请求成功、512/512 收到 SSE `[DONE]`；SLO 由 512 条记录逐条复算（`ttft_s ≤ 2.0 and tpot_s ≤ 0.5`）。

## 2. 与 v3 同表对比（唯一变量：sdpa → triton_paged）

| 负载档 | v4 SLO% (TPOT p95) | v3 SLO% (TPOT p95) | TPOT p95 变化 | v4 TTFT p95 | v3 TTFT p95 | v4 goodput | v3 goodput |
|---:|---|---|---:|---:|---:|---:|---:|
| 50% | **100.0%** (0.201 s) | 100.0% (0.229 s) | **−12.1%** | 0.780 s | 0.816 s | 0.428 | 0.427 |
| 75% | **100.0%** (0.290 s) | 100.0% (0.348 s) | **−16.7%** | 1.060 s | 1.073 s | 0.663 | 0.656 |
| 90% | **100.0%** (0.257 s) | 100.0% (0.312 s) | **−17.7%** | 1.043 s | 1.126 s | 0.711 | 0.706 |
| 100% | **100.0%** (0.327 s) | 100.0% (0.401 s) | **−18.4%** | 1.203 s | 1.270 s | 0.776 | 0.771 |

- 四档 SLO 保持 100%，TPOT p95 全面下降；100% 档 goodput +0.6%（0.771→0.776）。
- 降幅（−12%…−18%）落在离线预期的 **−20~−31%** 的下沿：在线是 64-token 输出 + 混步，paged 只作用于 decode；prefill 仍是 SDPA，因此混步里的 prefill 部分吃不到这部分收益。
- 90% 档 TPOT（0.257）低于 75% 档（0.290）的非单调与 v3 一致，来自 stall_free 的并发/混步差异（最大并发 18 vs 23），不影响结论。

## 3. 冻结门（跑前写入 `/root/p0_l5_gate.txt`，跑后未改）

| # | 门 | 实测 | 判定 |
|---|---|---|---|
| 1 | 四档全部 128/128 且收到 `[DONE]` | 4×128/128，512/512 `[DONE]` | ✅ |
| 2 | 100% 档 SLO = 100% | 100.0% | ✅ |
| 3a | 100% 档 TTFT p95 ≤ 1.33 s | 1.203 s | ✅ |
| 3b | 100% 档 TPOT p95 ≤ 0.42 s | 0.327 s | ✅ |
| 3c | 100% 档 goodput ≥ 0.73 req/s | 0.776 | ✅ |

**结论：L5 通过，`adopt_triton_paged` 正式转正。** `nanovllm/config.py` 与 `api_server.py` 的默认值已改为 `triton_paged`；sdpa 保留为显式 opt-in 与历史参考后端。

## 4. paged attention 与 KV cache 现状（一次说清）

| 项 | 现状 |
|---|---|
| 实现 | 自研 Triton：`nanovllm/kernels/paged_attention.py`，online softmax + fp32 累加，一个 program 对应 `(seq, q_head)` |
| 启动配置 | `BLOCK_N=32, num_warps=4, num_stages=1` 写死，**无 autotune**（E1 待做） |
| 覆盖范围 | **decode-only**；prefill 恒走 SDPA dense gather（`attention.py:124-145`） |
| 页大小 | 固定 **256 token**；kernel 硬约束 `head_dim==128`、`page_size==256` |
| KV 形状 | `[2, layers, blocks, 256, kv_heads_per_rank, 128]`，bf16（继承模型 dtype） |
| 本发 KV 实测 | rank0 **730** blocks / 8.55 GiB，rank1 **728** / 8.53 GiB；每块 **12,582,912 B = 12.0 MiB**；容量 ≈ **186,880 token/rank** |
| KV 管理 | xxhash 前缀缓存（按整 256-token 块）、ref_count 共享、free-list 分配；**无 swap/offload/淘汰/COW** |
| KV 量化 | **无**。模型名里的 FP8 是权重 FP8，加载期已反量化为 bf16；KV 与 kernel 均为 bf16（已登记 I20） |

## 5. 上一轮"paged 启动挂死"的真相：不是挂死，是无日志的慢加载

本轮给 `ModelRunner.__init__` 加了 `NANOVLLM_TRACE_INIT=1` 分段打点，并开 `faulthandler.dump_traceback_later(180)`。实测时间线（同一份代码、paged 启动）：

| 阶段 | 耗时 |
|---|---:|
| `dist.init_process_group` | 0.5 s |
| 建模（GPU 参数已分配，29.5 GiB/rank） | 9.1 s |
| **`load_model`（FP8 逐 expert 反量化）** | **351.4 s（rank0）/ 354.0 s（rank1）** |
| `warmup_model` | 16.4 s |
| `allocate_kv_cache` | 0.08 s |
| 合计到 `/health` 可用 | **376.9 s ≈ 6m17s** |

- faulthandler 在 3/6 分钟两次抓到的栈都是 `loader.py:103/108`（`safe_open` + `get_tensor` 的 FP8 反量化循环），**不是** barrier、不是 Triton 编译、不是 paged 路径。
- 代码层面也早就表明 paged 不可能影响启动：`warmup_model` 只跑 prefill（恒 SDPA），paged 模块在首次 decode 才惰性 import，`allocate_kv_cache` 无 backend 分支。
- 上一轮的现场（两 rank 各 29.5 GiB、零日志、0% util）与该阶段完全吻合：**权重已分配到 GPU 但 `load_model` 尚未返回**。冷页缓存下加载 13–15 min，15 分钟窗口自然等到"未起"。
- **运维结论**：启动窗口按 **≥20 min** 设（冷缓存），并用 `NANOVLLM_TRACE_INIT=1` 判活；不要再用"GPU 显存已占"推断"权重已加载完"。

## 6. 复现命令

```bash
cd /root/moe-server-s4000 && export PYTHONPATH=/root/moe-server-s4000
MODEL=/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8
# 本轮用显式 flag；转正后同一命令可省略 --attention-backend（默认已是 triton_paged）
python3 -m nanovllm.entrypoints.openai.api_server --model "$MODEL" \
  --served-model-name qwen3-30b-a3b --host 127.0.0.1 --port 8000 \
  --scheduler-policy stall_free --decode-moe-backend bmm_dense \
  --attention-backend triton_paged > /root/p6d_paged_server.log 2>&1 &
python3 scripts/benchmark_online.py --model qwen3-30b-a3b --tokenizer "$MODEL" \
  --max-qps 0.87 --requests 128 --max-tokens 64 --prompt-words 128 \
  --arrival-pattern poisson --slo-ttft-ms 2000 --slo-tpot-ms 500 \
  --fractions 0.5,0.75,0.9,1.0 --output /root/online-v4-paged.json
```

## 7. 局限

- 单实例/单客户端/固定形状（137-in/64-out）；SLO 按请求级 TPOT 而非逐 token ITL。
- 0.87 QPS 是 v2/v3 口径的封顶；四档全绿不等于 0.87 QPS 是饱和点。
- 本轮服务端开着 `NANOVLLM_TRACE_INIT=1`（faulthandler 每 3 min dump 一次主线程），对测量影响在微秒级，但为诚实记录在此声明。
- 跨后端不做硬门：paged 的参考基线为 `benchmarks/p2-reference-8tok-paged.json`，sdpa 参考仅对 sdpa 有效（`docs/P6-numerics-policy.md` §4）。
