# P29 Day1：平台/拓扑探测报告

> 日期：2026-11-09。范围：P29-A 平台能力 + 当前生产栈拓扑基线；不改生产默认。
> 原始 JSON：`benchmarks/data/P29/quant_probe.json`、`benchmarks/data/P29/topology2.json`（`benchmarks/data/` 不入库）。
> 复现脚本：`scripts/p29_probe_quant.py`、`scripts/p29_probe_topology.py`。

## 1. 今天做了什么

- 新增两个只读探针：量化 platform probe（不加载模型）和 30B 拓扑 profile（一次加载）。
- 给 `bmm_dense` decode 路径补了 `moe.local_pairs` 观测：此前只有 grouped prefill 路径有专家负载观测，decode profile 的 imbalance 恒为 0，属于观测盲区。观测只在 profiler 开启时执行，生产默认无影响。
- 没有改任何生产默认；30B 共加载两次（首次拓扑探针 + 补 `bmm_dense` 观测后的重跑），最终数据用第二次；结束后双卡 0 MiB。

## 2. 平台/量化探测（quant_probe）

环境：torch 2.11.0.post1 + torch_musa 2.11.0.post1+b431668，MTT S4000 47.9GiB，triton 3.2.0。

| 项 | 结果 |
|---|---|
| `mate` | 未安装 |
| `deep_gemm` / `deep_ep` | 未安装 |
| `_int_mm` | 存在，结果精确（64×256×64 max_abs_diff 0） |
| `_int_mm` 速度（2048²） | 比 bf16 慢 **5.0–57×**（M32–1024；speedup 0.0175–0.1994） |
| `_weight_int8pack_mm` W8A16（2048²） | 比 bf16 慢 **4.4–31×**（M1–256；speedup 0.0316–0.2272） |
| `_scaled_mm` FP8 | 报错 `scaled_gemm is only supported for MUSA_ARCH >= 310` |
| `_weight_int4pack_mm` / `_convert_weight_to_int4pack` | 符号存在；未做有效 4-bit 布局/质量/性能验证 |

**结论**：

- 当前 ATen 的 INT8/W8A16 路径不是可用的加速路径；S4000 的 INT8 200 TOPS 需要更好的 kernel（MATE 或自定义 fused dequant）才能兑现。
- FP8 运行时不可用；这与代码里 FP8 cast no-op 的历史观察一致。
- 8/4-bit 的第一优先级不是「量化整个模型」，而是先拿到 **FP8 权重保留 + fused dequant** 的显存收益，再评估 W8A16，最后 W4。

## 3. 当前生产栈拓扑 profile（30B，一次加载）

配置：TP2/EP2，`grouped_mm` prefill + `bmm_dense` decode，`triton_paged`，`stall_free`，chunk 256，eager，`max_model_len=140`，`gpu_memory_utilization=0.85`，profiler 开启；随机 token prompt。

| 阶段 | batch×context | clean step ms | profiled step ms | EP all-reduce / forward | rank-pair imbalance | attention / forward | MoE compute / forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| prefill | 1×128 | — | — | 2.58% | 8.87% | 1.37% | 89.68% |
| prefill | 8×128 | — | — | 4.95% | 8.36% | 8.79% | 67.35% |
| prefill | 32×128 | — | — | 7.82% | 8.55% | 15.57% | 50.89% |
| decode | 1×128 | 163.18 | 186.96 | 3.87% | 5.60% | 42.91% | 28.28% |
| decode | 8×128 | 183.63 | 200.88 | 2.99% | 4.20% | 3.26% | 64.07% |
| decode | 32×128 | 242.54 | 253.92 | 2.93% | 4.67% | 5.11% | 68.29% |

说明：

- `profiled step ms` 是 device-event profiler 打开时的步时；`clean step ms` 是同一进程里 `stop_profiling()` 后的 4 步。**不要和历史 142ms 直接混比**（本 harness 的 max_model_len/gpu-util/context 不同，且同进程已加载过 profiler）。
- 观测到的 decode imbalance 是补了 `bmm_dense` 观测后的值；此前 decode 实际为 0（没有观测，不是真的平衡）。
- prefill imbalance 约 8.4–8.9%，高于历史 b32/长上下文 profile 的 5.19%；说明 **placement/load balance 是当前 2 卡上真实可见的缺口**。
- EP all-reduce 占比：prefill 2.6–7.8%，decode 2.9–3.9%；在这组 128 context 负载上不是主瓶颈。

## 4. 对路线图的影响

1. **量化优先级不变但更明确**：先做 FP8 保留 + fused dequant；ATen INT8/W8 暂不作为加速候选；MATE 先探测安装/可用性。
2. **拓扑**：优先做 **EPLB/专家放置**（prefill imbalance 8.5% 比 all-reduce 更显眼）和 **量化后的 DP=2**；PP/SP/CP 继续不做。
3. **解码观测**：后续所有 EP/placement 实验必须用补过 `bmm_dense` 观测的 profile，否则会把 decode imbalance 看成 0。

## 5. 下一步（Day2）

- P29-B：FP8 权重保留 + fused dequant 单层原型（先不 e2e）。
- 同时探测：MATE 是否能在当前 torch_musa 上安装/调用；若不能，直接走 Triton dequant。
- 门：单层数值误差、单层步时/显存；不过门就回退，不进入 e2e。