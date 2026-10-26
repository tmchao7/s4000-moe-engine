# P6 — MUSA Graph capture：**可捕获**（推翻旧结论，路径已定）

> 旧版本（commit `ef60c3e`）的结论是"decode 不可捕获，P6 取消"。**那个结论有三处错误**，本文按逐算子实测重写。
> 证据：`benchmarks/data/p6_graph_probe.json`（每个 case 独立进程跑，含错误链）。
> 上游做法与出处见 `docs/P6-musa-graph-survey.md`。

## 1. 结论先行

| 问题 | 答案 |
|---|---|
| MUSA Graph 是 CUDA Graph 的等价物吗？ | 是。`torch.musa.MUSAGraph` / `torch.musa.graph(pool, stream, capture_error_mode)` / `graph_pool_handle` 与 CUDA 一一对应；本机 `torch 2.11.0.post1` 上 `torch.musa.is_current_stream_capturing()` 可用。 |
| 是 MoE 模型特性 / token 数未知导致不能捕获吗？ | **不是**。摩尔官方 S4000 文档写明 MUSA Graph 默认开启；厂商 MoE 代码里有 `is_current_stream_capturing()` 分支；本机实测厂商 Triton MoE 在 decode 各桶（1/2/8/32/64 token）都能捕获且数值完全一致。token 数可变用"按桶捕获 + padding"解决，本仓库 `graph_bs` 已是该机制。 |
| 现在能捕获 decode 了吗？ | **能**：图内 = Triton paged attention + 厂商 fused MoE + MCCL all-reduce，全部实测通过。故默认仍是 `enforce_eager=True`，但已有明确的、可 A/B 的候选路径（`decode_moe_backend`）。 |
| 还有什么不能捕获？ | `torch._grouped_mm`（我们默认 MoE 后端）、`torch.bincount`、任何主机同步（`.item()/tolist()/bool(t.any())`）。 |

## 2. 旧结论的三处错误（必须记住，否则会重复浪费）

1. **探针本身有 bug**：旧 `tests/p6_graph_probe.py` 把 4 个用例放在**同一个 `try` 块**里，第一个异常（`torch._grouped_mm`）之后 `paged` 用例**根本没执行**。文档里"Triton paged 也不可捕获"是**未验证的推断**。→ 新探针每个 case 独立捕获、独立报错；实际运行时再**每个 case 单独起进程**（一次失败的捕获会把流状态搞坏，后续 case 会收到"operation failed due to a previous error during capture"这种**掩盖型报错**）。
2. **我们自己的探针在捕获区内做了主机同步**：`paged_decode_attention(..., int(context_lens.max()))` 的 `int(...)` 在 `with graph:` 里执行 → 直接违规。真实引擎传的是主机侧的 `context.max_context_len`，所以这是**探针的错**，不是内核的错。修掉后 `triton_paged` **PASS**。
3. **`torch.bincount` 不可捕获**（报 `operation not permitted`）——它需要主机侧尺寸信息。我们 `_forward_grouped_mm` 里用它算 per-expert offsets，所以 `grouped_mm` 路径即便算子可捕获，这段也要改写成 `zeros(E).index_add_(ids, ones)`。本轮改用厂商 MoE 绕开。

## 3. 逐算子实测（每 case 独立进程，2×S4000，torch 2.11.0.post1 / triton 3.2.0 mtgpu / 驱动 5.2.0）

协议：建静态缓冲 → eager 写一次 → 捕获 → **改输入** → replay → 与 eager 重算比对；
`stale_graph=false` 表示 replay 确实读了新输入（不是重放旧结果）。

| case | 结果 | 数值 | 说明 |
|---|---|---|---|
| `plain` | ✅ | max_rel 0.0 | 基线 |
| `triton_copy` | ✅ | 0.0 | Triton 内核本身可捕获 |
| **`triton_paged`** | ✅ | 0.0 | 我们的 paged decode kernel（b4/h16/d128/page256，runtime context_len）**可捕获** |
| `sdpa_decode` | ✅ | 0.0 | SDPA decode 也可捕获（但 kv_len 必须固定，见 §5） |
| `bmm_batched` | ✅ | 0.0 | 批量 GEMM 可捕获（可捕获 MoE 的备选实现） |
| **`vendor_fused_moe`** | ✅ | 0.0 | 厂商 Triton fused MoE，E=64/H=2048/N=768，32 token，mp22 关闭 |
| **`vendor_fused_moe_ep`** | ✅ | 0.0 | 同上但带 **EP2 `expert_map`**，且 **1/2/8/32/64 token 每个桶各一张图**全部通过 |
| `mccl_all_reduce` | ✅ | 0.0（两 rank） | 每层一次 EP all-reduce 可入图 |
| `aten_grouped_mm` | ❌ | — | `torch._grouped_mm`：`operation not permitted when stream is capturing` |
| `aten_grouped_mm_relaxed` | ❌ | — | 换 `capture_error_mode='relaxed'` 无效 |
| `route_pack` | ❌ | — | `torch.bincount` 是卡点（其余 cumsum/argsort/index_select/index_add_ 未单独验证，见 §5 待办） |
| `host_sync` | ❌（预期失败） | — | `bool(tensor.any())`：这正是我们修掉的旧代码缺陷 |

**决定**：decode 图 = `triton_paged` + `vllm_fused`(厂商 MoE) + MCCL，prefill 仍 eager。这是**分支 A**（见 `docs/P6-musa-graph-survey.md` §G）。

## 4. 为此改的代码（已经在仓库里，默认行为不变）

| 改动 | 作用 |
|---|---|
| `nanovllm/utils/compat.py::is_capturing()` | 统一封装 `torch.musa.is_current_stream_capturing()`（照厂商写法） |
| `nanovllm/layers/attention.py::store_kvcache` | `(slot_mapping < 0).any()` 是主机同步；捕获时走无分支的 masked 写入 |
| `nanovllm/models/qwen3_moe.py` | `bool(valid.any())` / `bool(mask.any())` 早退在捕获时跳过（否则必然失败） |
| `nanovllm/config.py::decode_moe_backend` | 默认 `None`；设成 `vllm_fused` 时**只有被捕获的 decode 图**用它，eager prefill 仍用 `moe_backend`（`grouped_mm`），不牺牲 TTFT |
| `nanovllm/engine/model_runner.py` | 捕获时临时切 MoE 后端、捕获后还原；打印桶/显存；新增 `set_enforce_eager()`（同进程 A/B）与 `graph_report()`/`memory_report()` |
| `scripts/p5_graph_ab.py` | 一次加载内跑 3 种配置：`eager_default`（今天默认）/ `eager_candidate`（同内核，eager）/ `graph_candidate`（同内核，replay），并做 token 一致性 |

## 5. 仍不能进图的东西（写代码时必须绕开）

1. **主机同步**：`.item()` / `.tolist()` / `bool(tensor.any())` / `if tensor:`。**注意 `for i in range(len(cu_q)-1)` 这类按数据长度的 Python 循环也只适合 prefill（prefill 不进图）**。
2. **`torch.bincount`**：改用 `torch.zeros(num_experts, dtype=int32).index_add_(ids, torch.ones_like(ids))`（本轮未单独验证，属待办）。
3. **`torch._grouped_mm`**：不可捕获（MUDNN fill 路径）。要用图就必须换 MoE 实现（厂商 MoE / bmm 稠密 / 自研 Triton）。
4. **SDPA decode 的 `kv_len` 是主机 int**：捕获时会被冻结成 `max_blocks*block_size`（本项目 4096），意味着短上下文也要按 4096 算 attention → 慢 8 倍。**所以图内 attention 必须用 `triton_paged`**（它从设备张量读 `context_lens`）。这条是"为什么不能简单把 sdpa 包进图"的关键。
5. **`torch.compile` / JIT 首次编译**：捕获前必须 warmup 一次（`capture_cudagraph` 已对每个桶做 warmup forward）。

## 6. 验收方法与冻结门（`scripts/p5_decide.py --mode graph`）

- 门槛（写死，禁止看结果再改）：token 完全一致、b1 吞吐 ≥ +15%、b8 ≥ +5%、b32/b64 ≥ 0.98、CV ≤ 5%、显存增量 ≤ 2 GiB。
- 跑法（一次加载，同进程切换）：

```bash
cd /root/moe-server-s4000 && export PYTHONPATH=/root/moe-server-s4000
python3 scripts/p5_graph_ab.py --batches 1,2,4,8,16,32 --repeats 3 \
  --decode-moe-backend vllm_fused --prefill-moe-backend grouped_mm \
  --attention-backend triton_paged --output /root/p5_graph_ab.json   # > /root/p5_graph_ab.log 2>&1
python3 scripts/p5_decide.py --mode graph --input /root/p5_graph_ab.json \
  --output benchmarks/p5-graph-ab-decision.json --markdown /root/p5_graph_ab.md
```

- **必须看的两个独立性**：
  1. `graph_candidate` vs `eager_candidate`：**同一批内核**，只差"有没有图"→ 这个比值才是图捕获的收益/正确性门。
  2. `eager_candidate` vs `eager_default`：换内核（`sdpa`→`triton_paged`，`grouped_mm`→`vllm_fused`）的代价；如果 vllm_fused 比 grouped_mm 慢很多，图捕获必须把差距赚回来，否则不采纳。

## 7. 若 A/B 失败的回退路线（按顺序，不重复踩坑）

1. **只捕小桶**：b1–8 收益最大；把 `graph_bs` 裁到 `[1,2,4,8]`，大桶继续 eager。
2. **换更快的可捕获 MoE**：`bmm_batched` 已实测可捕获；实现"激活专家 gather + padded bmm"（MegaBlocks 的问题分解方式），decode 用图、prefill 用 `grouped_mm`。
3. **部分捕获（SGLang breakable graph 方案）**：把不可捕获段挤出图外，其余照捕（机制见 `docs/P6-musa-graph-survey.md` §D）。
4. **等 runtime 修 `torch._grouped_mm`**：把 §3 的失败清单原样提 issue（含 torch/triton/driver 版本 + 三行最小复现）。

## 8. 复现探针

```bash
# 每个 case 一个进程（关键！失败的捕获会污染流）
for c in plain triton_copy triton_paged sdpa_decode bmm_batched route_pack host_sync \
         aten_grouped_mm aten_grouped_mm_relaxed vendor_fused_moe vendor_fused_moe_ep; do
  PYTHONPATH=/root/moe-server-s4000 python3 tests/p6_graph_probe.py --cases $c \
    --verbose --output /root/p6_case_$c.json
done
PYTHONPATH=/root/moe-server-s4000 python3 tests/p6_graph_probe.py \
  --cases mccl_all_reduce --world-size 2 --output /root/p6_case_mccl.json
```

`P6_VENDOR_MP22=1` 可额外测试厂商 mp22 fast path（历史上在 1024 token 触发 illegal memory access，默认关闭）。
