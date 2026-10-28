# P6 — I19 每步固定成本：逐层计时（launch-bound 判定）

> 日期：2026-10-27 19:21 / 19:44（两次同参重复，第二次修正了"每次 launch"指标）。
> 模型：`Qwen3-0.6B`（dense，28 层），**TP1**（排除 MCCL 干扰），`enforce_eager=True`、`attention=sdpa`、`fcfs`、b1、prompt 128 / 32 decode token。
> 脚本：`tests/p6_layer_timing.py`；证据：`benchmarks/data/p6d/p6_layer_timing_06.json`（sha256 `ec7e02078d8bb9695737f91c5bb7f9fa8739a4259ff7119f740806db9377eef3`）。
> 门（跑前固定）：`device_busy = kernel 设备时间 / 步时`；`<0.70` = launch-bound，`≥0.90` = 单 kernel 慢，中间 = 兼有。

## 1. 测法

同一进程、同一份权重，四段测量：

| 段 | 做法 | 回答 |
|---|---|---|
| baseline | 不装 hook，跑 30 个 decode step | 步时基线 |
| async | 28 个 decoder layer 装 forward_pre/post hook，只在 decode step 采集 | 每层 **host 发射时间** |
| sync | 每层结束 `torch.musa.synchronize()` 后取样 | 每层 **设备跨度**（含层内 kernel + 发射间隙） |
| profile | `torch.profiler` 跑 1 个 decode step | kernel 类型数、总 launch 数、设备时间、top-15 |

## 2. 结果（两次独立重复）

| 指标 | 第一次 | 第二次（修正指标后） |
|---|---:|---:|
| decode 步时（中位） | 61.18 ms | **53.74 ms** |
| 每层 host 发射（中位） | 2.127 ms | 1.842 ms |
| 每层设备跨度（中位） | 2.254 ms | 1.996 ms |
| host × 28 | 59.57 ms | 51.58 ms |
| 设备跨度 × 28 | 63.13 ms | 55.89 ms |
| kernel 设备时间合计 | 29.27 ms | **32.42 ms** |
| 每步 launch 总数 | ≥1,632（旧脚本只存 top-15） | **2,401** |
| kernel 类型数 | 35 | 35 |
| 平均每次 launch 设备时间 | ~18 µs | **13.5 µs** |
| **device_busy = kernel / 步时** | **0.478** | **0.603** |
| 判定 | **launch_bound** | **launch_bound** |

两次重复的步时差 ~7 ms（53.7 vs 61.2），但 kernel 设备时间稳定在 29–32 ms，`device_busy` 0.48–0.60，**都低于 0.70 的门**，结论一致。

## 3. 怎么读

1. **CPU 发射时间 ≈ 整步时间**：28 层 × 1.84 ms = **51.6 ms**，步时 **53.7 ms**。也就是说，CPU 把这一层的 kernel 发完的时间，基本就是这一层在墙钟上的时间——GPU 没有机会跑到前面去。
2. **GPU 有 40% 时间空转**：一步只执行了 32.4 ms 的 kernel，墙钟 53.7 ms。空转不是算力不够，是**供不上**。
3. **2,401 次 launch / 步 = 每层 ~86 次**，平均每次只有 **13.5 µs** 的设备时间。这是"小 kernel 洪水"，不是"某个 kernel 特别慢"。
4. **top kernel 的构成**（第一次 profile，设备时间合计 21.6 ms = 74%）：
   - `CastOp` fp32↔bf16（225 + 226 次）→ RMSNorm/残差在 fp32 里做；
   - `Mul` + 224 次、`Add`、`Rsqrt`、`Pow`（各 112–113 次）→ RMSNorm 与 RoPE；
   - `SelectOp` 56 次 + `Concat` 56 次 → SDPA decode 的 mask/gather 拼装；
   - `IndexVectorizedKernel` 56 次 → KV 写入；
   - 真正的 flash attention 只有 **28 次 / 2.3 ms**；GEMV/GEMM（`batch_gemv_row_continuous`）合计 **~3.4 ms**。
   
   即：**算力只占一小部分，开销集中在 norm/rope/mask 这些逐元素小 kernel 上**。

## 4. 判定与下一步

**I19 = launch/CPU-bound，不是单 kernel 慢。** 三条推论：

1. **图路径（I1/I2）的收益是真实的**：同模型 eager 63.2 ms vs graph replay 40.0 ms 的 23.2 ms 差值，与"GPU 空转 ~20–24 ms"的测量吻合。但图被 replay 漂移卡住，**不能作为近期收益**。
2. **不进图也能拿一部分**：`2,401` 次 launch 里大部分来自 RMSNorm（fp32 上采样 → Cast/Pow/Rsqrt/Add/Mul）与 RoPE。把它们融合成 1–2 个 kernel（或在 bf16 内完成）可以消掉每层几十次 launch，且不依赖图。这是 I19 的新可执行方向（纯 eager、可逐项 A/B）。
3. **对 30B 的映射**：30B b1 = 175 ms、forward 占 95.5%，每层 ~2.2 ms 的结构与 0.6B 同源；30B 的绝对 launch 数更多。若融合 norm/rope 能在 0.6B 上拿 10 ms 量级，30B 的收益应按层数放大（需单独实测，不能直接外推）。

## 5. 局限

- 0.6B TP1 与 30B TP2 不是同一栈：TP1 排除了 MCCL，结论"launch-bound"对 TP2 仍需一次抽查（MCCL 只会增加每层固定成本，不会让它变成 kernel-bound）。
- profiler 本身把步时放大到 ~0.70 s（11×），所以**只用它的设备时间与 launch 计数，不用它的墙钟**。
- sync 段强制串行，设备跨度含发射间隙，因此"设备跨度 × 28"略大于真实步时（55.9 vs 53.7 ms）——方向正确。
