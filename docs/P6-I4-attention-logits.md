# P6 I4 — Triton paged attention logits 诊断（2026-10-27）

> **结论**：I4 已完成定位，**没有发现页表、mask 或 poison-tail 地址错误**；差异是 SDPA 与 Triton 在线 softmax / bf16 累加路径的可复现数值偏差，经 48 层模型放大后会翻转少数近平局 token。冻结门未过，`attention_backend` 继续默认 **sdpa**。没有内核补丁、没有第二次 30B 验证加载。

## 方法

- 代码：`41ab19b` 新增 `ModelRunner.probe_attention_backends()` 与 `scripts/p6_attention_logits.py`；TP2/EP2、`grouped_mm`（与冻结 attention gate 同口径）、`enforce_eager=True`。
- 每个 decode step 先快照当前槽位 KV，跑 SDPA 并保存其 KV 写入；还原后跑 `triton_paged`；最后恢复 SDPA KV 并采样 SDPA token。后续步骤因此始终是 **SDPA teacher-forced**，不会把早期候选 token 分歧级联放大。
- rank 0 在完整 vocab logits 上计算精确 `max_abs`、top-1/2 margin 和按层 attention 输出差；rank 1 正常参与 TP 通信。四个既有 workload（16/32 × 512/2048）、32 output、3 repeats、seed `20261026`。

## 结果

低层扩展回归（255/256/257、511/512/513、2047/2048/2049，混合长度、乱序物理页、poison tail）通过：`max_abs=1.2207e-4 < 2e-3`，无 NaN。30B 探针的 12 次 run 也全有限：

| workload | 每 repeat 的 top-1 翻转 / 比较数 | 最大 logits 差 | 最大层输出差 |
|---|---:|---:|---:|
| b16/c512 | 1 / 496（3/3 repeat 相同） | 3.9766 | 7.5588 |
| b32/c512 | 1 / 992（3/3 repeat 相同） | 4.0586 | 8.7217 |
| b16/c2048 | 0 / 496 | 1.4688 | 1.0625 |
| b32/c2048 | 2、6、5 / 992 | 6.0938 | 6.6107 |

- 合计是 **19/8,928** 同状态 request-step top-1 翻转；这不是冻结门的请求级分母，不能用来替换既有 `0.875 < 0.99` 的 A/B 判决。
- 首层 attention 输出的全局 `max_abs` 只有 **2.4414e-4**，而最大差在 layer 36 达到 **8.7217**；这是小的底层数值误差逐层放大的形态，不是页边界/掩码读错。
- 所有翻转的 baseline top-2 margin 都在 **0–0.3125**；例如 b16/c512 的固定首步翻转 margin 是 0.0625。它们是 greedily sampled 近平局，而非 NaN 或大范围错误。

## 决策

冻结门和默认值均不动。由于直接参考、边界矩阵和首层诊断都没有给出单一可修的内核错误，不能以试错方式“模仿 SDPA”而牺牲 Triton 的收益；因此不使用第二次 30B 加载重跑 A/B。若未来产品要接受不同但稳定的 greedy 输出，必须先由产品方明确新的正确性契约并另立门禁；在此之前保持 SDPA。

## 证据

- 服务器：`/root/p6_attention_logits_i4.json`（`f9b686bcb46e768d6…`）、`/root/p6_attention_logits_i4.jsonl`（`46a06091ae7df233…`）。
- 本地忽略归档：`benchmarks/data/p6b/p6_attention_logits_i4.json` 与 `.jsonl`，sha256 同上。
- 低层门：服务器 `PYTHONPATH=. python3 tests/p5_paged_attention.py` 输出 `boundary_max_abs=0.00012207`；`tests/p5_decide.py` 通过。
