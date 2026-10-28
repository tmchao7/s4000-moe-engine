# P2 — Qwen3-30B-A3B-FP8 双卡（TP2 · EP2）正确性与性能

> 日期：2026-10-26　硬件：2× MTT S4000（48GB，PCIe Gen5 x16）　软件：torch 2.11 + torch_musa 2.11、MCCL 2.11.4、transformers 5.13、Python 3.10
> 模型：`Qwen3-30B-A3B-Instruct-2507-FP8`（加载期 e4m3+block scale → bf16 反量化）　并行：Attention TP2 + 专家 EP2（每卡 64 专家）

## 结论（TL;DR）

1. **packed experts + `torch._grouped_mm` 相对 loop 后端稳定快 3.0–3.2×**（batch 16–64），legacy p95 ITL 降到 0.31×；batch 1–8 也有 2.67–2.80×。
2. **batch 32：74.0 tok/s（loop 24.2）、legacy p95 ITL 0.404s、p95 TTFT 1.52s；batch 64：120.4 tok/s、legacy p95 ITL 0.477s、p95 TTFT 2.49s**，旧口径 CV ≤ 1.1%。
3. 吞吐随 batch **近似线性**（1.27→39.8 tok/s loop；3.38→120.4 tok/s grouped），说明瓶颈已从"每专家 Python 循环 + GPU 同步"转为 GEMM/带宽，PCIe TP 通信未成为限制。
4. **正确性门**：tiny 模型 TP1/TP2 × loop/grouped 四路逐 token 一致；30B 四组中英文 prompt 输出连贯；30B 上 loop 与 grouped **4/4 条 prompt 逐 token 相同**，且**同一后端 3 次重复 + 首次调用全部一致**（本轮修掉了 grouped 的原子累加不确定性，见下）。
5. packed 权重布局在加载期每卡约 **29.5 GiB**，比旧的 64 个独立专家模块（~32.8 GiB）少约 **3.3 GiB**，且没有为 grouped GEMM 额外复制一份权重。

## 1. 本轮修正：grouped-MM 的贪心不可复现

**现象**：同一进程内重复生成同样的 4 条 prompt（temperature=0），`grouped_mm` 会偶尔给出不同的 token（发生在近并列处，例如"\n\n\n" 之后选 "```" 还是 "d"），而 `loop` 后端 3/3 稳定；`grouped_mm` 更换后端后的**第一次**前向也与稳态不同。

**根因**：`_forward_grouped_mm` 用 `out.index_add_(0, flat_token, contribution)` 合并 top-8 贡献。同一个 token 的 8 个 slot 会命中同一行 → 重复索引 → MUSA 上用原子累加，**累加顺序由 kernel 调度决定**，bf16 下几个 ulp 的差异足以翻转近并列 argmax。`loop` 后端每个专家内 token 索引唯一，不触发原子累加，所以确定。

**修复**（`nanovllm/models/qwen3_moe.py`）：把贡献先无累加地 scatter 回 `(token, slot)` 布局，再沿 slot 维按固定顺序求和 ——
`dense.view(num_tokens * top_k, -1)[flat_slot] = contribution; return dense.sum(dim=1)`。
代价是最坏情况（16k token prefill）多一块 ~512 MiB 的临时缓冲；等价性与确定性由 tiny 四路 + 重复测试守住。

**修复后的证据**（`/root/p2_ablation_full.json` 的 `verification` 字段）：

```
loop       deterministic=True  warmup_matches_steady=True
grouped_mm deterministic=True  warmup_matches_steady=True
grouped_mm vs loop prompt matches=[True, True, True, True]
```

> ⚠️ 因此旧的 `p2_outputs.json`（修复前的参考）已被 `benchmarks/p2-reference-8tok.json` 取代：减少顺序变化后，prompt 3（"请用 Python 写一个快速排序"）的贪心首 token 从 `\n\n\n```python` 变为 `当然可以！下面是一个带有详细注`。两者都是连贯输出，新的这份才是当前构建下**可复现**的答案。

## 2. 正确性门（分层，先过门再测性能）

| 层 | 测试 | 结果 |
|---|---|---|
| 逐层数值参考 | `tests/p2_ground_truth.py`（手写 PyTorch CPU 参考 vs tiny TP1） | 一致 |
| 并行等价 | `tests/p2_tp_check.py`（TP1==TP2） | 一致 |
| 后端/放置等价 | `tests/p2_grouped_mm.py`（TP1/TP2 × loop/grouped + round_robin + 重复确定性） | 五路 token 全一致 |
| 调度等价 | `tests/p3_scheduler.py`（FCFS vs stall_free，mixed step） | 输出一致，出现 3 个 mixed step |
| 服务层 | `tests/p4_api.py` / `tests/p4_async_engine.py` / `tests/p4_benchmark_client.py` | 通过 |
| 30B 端到端 | `scripts/p2_benchmark.py --verify-prompts`（4 条固定 prompt，2 后端 × 3 重复 + 预热） | 两端确定；跨后端 4/4 一致 |

30B 的 4 条 prompt 输出（修复后，greedy 8 token）：

- `简单介绍一下 MoE（Mixture of Experts）架构的核心思想。` → ` MoE（Mixture of Experts，`
- `What is paged attention in LLM inference? Answer in two sentences.` → ` Paged attention is a memory management technique`
- `1 到 100 的整数中所有质数之和是多少？给出答案即可。` → ` 1060\n\n解释：`
- `请用 Python 写一个快速排序，附注释。` → `当然可以！下面是一个带有详细注`

## 3. 离线批处理消融（loop vs grouped-MM）

方法：随机 token prompt（128 in / 32 out），每个点先 warmup 1 次再测 3 次，模型只加载一次、后端在进程内切换。本表冻结于 benchmark schema v1：脚本实际使用 `pstdev/median` 计算 CV，并把全部 token interval 打平后的 p95 记成了 TPOT；按当前规范应称 **legacy ITL**。原始 JSON 不篡改，P5 schema v2 已改为 `sample std/mean` 和逐请求 TPOT。

| Backend | Batch | Output tok/s | legacy CV | TTFT p95 (s) | legacy ITL p95 (s) | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| loop | 1 | 1.27 | 0.002 | 1.528 | 0.7722 | 1.00x |
| loop | 2 | 2.25 | 0.002 | 1.749 | 0.8705 | 1.00x |
| loop | 4 | 4.50 | 0.002 | 1.856 | 0.8737 | 1.00x |
| loop | 8 | 8.83 | 0.009 | 2.141 | 0.9126 | 1.00x |
| loop | 16 | 13.43 | 0.004 | 2.663 | 1.1842 | 1.00x |
| loop | 32 | 24.22 | 0.011 | 3.628 | 1.2984 | 1.00x |
| loop | 64 | 39.83 | 0.009 | 5.408 | 1.5468 | 1.00x |
| grouped_mm | 1 | 3.38 | 0.002 | 0.429 | 0.2946 | 2.67x |
| grouped_mm | 2 | 6.30 | 0.003 | 0.500 | 0.3138 | 2.80x |
| grouped_mm | 4 | 12.37 | 0.001 | 0.583 | 0.3181 | 2.75x |
| grouped_mm | 8 | 24.21 | 0.003 | 0.725 | 0.3226 | 2.74x |
| grouped_mm | 16 | 42.68 | 0.005 | 0.996 | 0.3593 | 3.18x |
| grouped_mm | 32 | 74.01 | 0.003 | 1.522 | 0.4037 | 3.06x |
| grouped_mm | 64 | 120.39 | 0.001 | 2.489 | 0.4765 | 3.02x |

### 验收门

| Batch | Speedup ≥1.5x | p95 legacy ITL ratio ≤1.10 | legacy CV ≤5% | Result |
|---:|---:|---:|---:|---|
| 16 | 3.18x | 0.303 | 0.005 | **PASS** |
| 32 | 3.06x | 0.311 | 0.003 | **PASS** |

![throughput](data/throughput.png)

![legacy itl p95](data/tpot-p95.png)

## 4. 交叉验证（两次独立运行）

同一配置在**两次独立进程**里各跑了一遍 batch 1/16/32（第一次是会话交接时遗留的旧版本，第二次是本轮修复后的版本）：

| Batch | loop 旧→新 (tok/s) | grouped 旧→新 (tok/s) |
|---:|---|---|
| 1 | 1.26 → 1.27 | 3.57 → 3.38 |
| 16 | 13.63 → 13.43 | 43.35 → 42.68 |
| 32 | 23.87 → 24.22 | 75.42 → 74.01 |

差异 ≤3%，在跨进程噪声范围内；确定性修复没有带来可测的性能代价（b32 74.0 vs 75.4 tok/s，同一量级）。

## 5. 显存

| 版本 | 每卡显存（权重加载后，未计 KV） |
|---|---|
| 旧：64 个独立专家 `nn.Module` | ~32.8 GiB |
| 新：单份 packed `[expert, in, out]` | **~29.5 GiB** |

节省 ~3.3 GiB/卡，且 grouped GEMM 直接消费同一份存储（不再复制第二份 packed 权重）。

## 6. 局限与下一步

- 注意力仍是 SDPA + 每步 block gather，未接 Triton-mtgpu paged attention kernel；decode 的 `k_cache[blocks]` gather 是下一步主要优化点。
- 未启用 CUDAGraph（`enforce_eager`），b1–b8 的固定开销里仍有 Python 调度与张量构造。
- grouped-MM 在小 batch 的固定开销（`argsort`/`bincount`/两次 `_grouped_mm`）使 b1–b8 的加速比（2.7–2.8×）低于 b16–64（3.0–3.2×）。
- 本轮吞吐测试用 `temperature=0`；与外部数字对比时需统一说明。
- **可复现性的边界（P5 补充）**：本报告的"同一后端稳态可复现"是在 4 条 prompt / batch≈4 / 8 token 下验证的。P5 的 attention A/B 进一步测到：**batch 32 的 decode 跨重复不可复现**，且 `sdpa` 与 `triton_paged` 都受影响（首次分歧在第 2 个 token），指向两者共享的 MoE grouped GEMM 或 MCCL all-reduce，而不是 attention 后端。在该问题定位前，b32 以上的逐 token 结论不可用，详见 `benchmarks/P5-attention-ab.md` §3。

## 7. 复现

```bash
# 消融（一次加载，含正确性门与确定性检查）
PYTHONPATH=. python3 scripts/p2_benchmark.py \
  --batches 1,2,4,8,16,32,64 --prompt-len 128 --output-len 32 --repeats 3 \
  --moe-backends loop,grouped_mm --scheduler-policy fcfs \
  --verify-prompts --reference-outputs benchmarks/p2-reference-8tok.json \
  --verify-repeats 3 --verify-backends loop,grouped_mm \
  --output benchmarks/data/p2_ablation_full.json

# 表格 + 曲线
python3 scripts/render_benchmark.py benchmarks/data/p2_ablation_full.json \
  --output-dir benchmarks/data --accept-batches 16,32
```
