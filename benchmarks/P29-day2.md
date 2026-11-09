# P29 Day2：FP8 保留 / W8A16 原型报告

> 日期：2026-11-09。范围：P29-B 的内存模型 + 两个单层原型；不改生产默认。
> 原始 JSON：`benchmarks/data/P29/int8_gemv.json`（`benchmarks/data/` 不入库）。
> 脚本：`scripts/p29_fp8_gemv.py`（FP8 负结果）、`scripts/p29_int8_gemv.py`（INT8 原型）。

## 1. 30B 权重/显存模型（checkpoint 实测）

直接枚举 4 个 safetensors 的 dtype/numel（不加载模型）：

| 项 | 数值 |
|---|---:|
| checkpoint 文件 | **29.03 GiB**（4 shards） |
| FP8 e4m3 权重 | 29.897B 参数 = **27.84 GiB**（1 byte） |
| bf16 非量化权重 | 0.637B 参数 = **1.19 GiB** |
| 总参数 | 30.53B |
| 反量化成 bf16 后 | **56.88 GiB**（29.897B×2 + 1.19） |
| TP2 每 rank（bf16） | 约 **28.4 GiB** + KV/激活（当前生产约 29.5 GiB packed） |
| TP2 每 rank（保留 FP8） | 约 **14.5 GiB** + scales/KV |
| 完整模型单卡（保留 FP8） | 约 **29.0 GiB**，48GB 可放下 |
| 完整模型单卡（bf16） | 约 **56.9 GiB**，放不下 |
| W8（int8）完整模型 | 约 **29.0 GiB**（与 FP8 同量级） |
| W4 完整模型 | 约 **15.1 GiB** + scales |
| KV（全模型） | 96 KiB/token；4k context 单序列约 384 MiB |

**结论**：量化/FP8 保留首先解锁的是 **显存/拓扑**：

- bf16 必须 TP2；完整模型单卡不可能。
- FP8/W8 完整模型约 29GiB，**单卡 48GB 可放**，才可能做 **DP=2 两个副本**（高吞吐）或更大 KV/上下文。
- W4 约 15GiB，单卡空间更大，但质量/内核风险最高。

## 2. FP8 保留 + fused dequant：Triton 负结果

原型：`scripts/p29_fp8_gemv.py`，FP8 e4m3 权重 + 每输出通道 scale，Triton GEMV 内 dequant。

结果：编译失败，MUSA linker 报：

```text
ld.lld: error: relocation R_MTGPU_REL32_DWORD cannot be used against
symbol '__mt_tt_e4m3_to_f32'; recompile with -fPIC
```

含义：**当前 Triton-mtgpu 不支持在 kernel 内做 e4m3→f32 转换**。

- 不能靠 Triton 写「保留 FP8 + kernel 内 dequant」；
- 备选：MATE 的 8-bit/FP4 路径（当前未安装）、手写 MUSA C++/汇编、或先 dequant 到 bf16 再 GEMM（省不了权重带宽/显存峰值）。

## 3. INT8 保留 + GEMV：原型可跑，但 M=1 仍慢于 bf16

原型：`scripts/p29_int8_gemv.py`，int8 权重 + 每输出通道 scale，Triton GEMV 内 dequant；K=N=2048，M=1。

| 指标 | 数值 |
|---|---:|
| bf16 权重 | 8.0 MiB |
| int8 权重 | 4.0 MiB |
| bf16 GEMV | 0.1216 ms |
| dequant 后 bf16 GEMV（参考） | 0.1230 ms |
| 自定义 int8 fused GEMV | **0.1606 ms** |
| 相对 bf16 | **0.76×（更慢）** |
| 数值误差（vs 参考） | max abs diff 0.5（量化误差，非 kernel bug） |

对比 Day1 的 ATen `_weight_int8pack_mm`（0.55 ms）：自定义 GEMV 已快约 3.4×，但仍比 bf16 慢。

原因判断：M=1 的 GEMV 不走 INT8 tensor core；kernel 里 int8→fp32 + 逐元素乘加是标量/FP32 路径，抵消了省带宽收益。

## 4. Day2 结论

1. **显存收益是真实的**：FP8/W8 完整模型约 29GiB，单卡可放；bf16 56.9GiB 不行。
2. **decode 速度收益目前不成立**：FP8 Triton 不可用；INT8 自定义 GEMV 仍 0.76× bf16；ATen int8 更慢。
3. **量化优先级要改成「先省显存/解锁 DP，再谈 decode 加速」**；不能把 4-bit 当性能银弹。
4. **下一步**：
   - 先探测 MATE 能否在当前 torch_musa 上使用（8-bit/W4A8 路径）；
   - 若不能，评估 W8 在 **prefill M>1** 或 **INT8 tensor core** 上的收益（需要 W8A8/激活量化，质量风险）；
   - 若仍无速度收益，量化只作为 DP/长上下文/容量特性，decode 保持 bf16。

## 5. 对 9 天计划的影响

- Day3 不再直接做 e2e W8；先做 **MATE 可用性 + W8 tensor-core 微基准**，不过门就切 EPLB/拓扑，不把时间沉进量化。
- DP=2 的内存前提已满足（FP8/W8 完整模型 29GiB）；但 DP 需要 DP Attention/AllToAll，当前引擎没有，属于更大的改动；先做单卡完整模型 + 两个 server 副本的可行性评估，而不是直接改 EP。
- 4-bit 继续往后放；在 W8 没有速度收益前不做。