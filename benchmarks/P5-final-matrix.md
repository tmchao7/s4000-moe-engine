# P5 — 终局离线矩阵（schema v2，默认栈）与前缀缓存探针状态

> 日期：2026-10-26　硬件：2× MTT S4000　模型：Qwen3-30B-A3B-Instruct-2507-FP8（TP2/EP2）
> 配置：`attention_backend=sdpa`、`moe_backend=grouped_mm`、`expert_placement=linear`、`scheduler_policy=fcfs`、`enforce_eager=true`
> 原始数据：`benchmarks/data/p5_final_matrix_partial.json`（由运行日志逐条恢复，见 §4）

## 1. 结果（prompt 128 / output 32，每点 warmup + 3 次，batch 64 为 2 次）

| batch | output tok/s | CV | TTFT p95 (s) | 请求级 TPOT p95 (s) | ITL p95 (s) | repeats |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.06 | 0.005 | 0.454 | 0.3230 | 0.3370 | 3 |
| 2 | 5.70 | 0.005 | 0.531 | 0.3448 | 0.3538 | 3 |
| 4 | 11.44 | 0.006 | 0.604 | 0.3414 | 0.3491 | 3 |
| 8 | 22.57 | 0.001 | 0.759 | 0.3414 | 0.3541 | 3 |
| 16 | 40.57 | 0.024 | 1.051 | 0.3732 | 0.3846 | 3 |
| 32 | 69.39 | 0.018 | 1.599 | 0.4244 | 0.4589 | 3 |
| 64 | 111.75 | 0.007 | 2.674 | 0.5048 | 0.5217 | 2 |

- 吞吐随 batch 近似线性（3.06 → 111.75 tok/s），CV ≤2.4%，无异常点。
- 与 P2 的 legacy v1（b32 74.0、b64 120.4 tok/s）同量级；**不可直接比**：v1 的 CV 是 `pstdev/median`、`tpot` 字段实为 token-gap ITL，且 v1 的 prompt/output 与本次相同但未记录请求级 TPOT。差异主要来自指标口径与 run-to-run 波动。
- TTFT p95 随 batch 上升（0.45 → 2.67s），因为 prefill 需要处理 128×batch 个 token；这是 FCFS 的预期行为，与 P3 的结论一致。

## 2. 与 attention A/B 的关系

`P5-attention-ab.md` 测到 `triton_paged` 在同一后端下比 SDPA 快 19%–59%，但由于 **batch 32 的 decode 在本栈上不可复现（两个后端都不可复现）**，冻结门判定 `keep_sdpa`。因此本矩阵使用 SDPA，代表**当前默认栈的最终数字**；paged kernel 的收益要等 b32 可复现性问题定位后才能落地。

## 3. 前缀缓存冷/热探针：未完成

原计划 `--probe-batches 8 --probe-prompt-len 512 --probe-shared-prefix-len 384 --probe-cache-modes cold,hot` 与矩阵同进程执行，但本次在**最后一个矩阵点（b64 第 3 次重复）**崩溃（`RuntimeError: MUSA error: unknown error` → rank1 退出 → `DistBackendError`），probe 阶段未运行。

复现命令（下一次加载）：

```bash
python3 scripts/p2_benchmark.py --batches 1,2,4,8,16,32 --prompt-len 128 --output-len 32 \
  --repeats 3 --moe-backends grouped_mm --attention-backends sdpa \
  --probe-batches 8 --probe-prompt-len 512 --probe-shared-prefix-len 384 \
  --probe-cache-modes cold,hot --output /root/p5_final_matrix.json
```
建议去掉 b64（疑似触发点），矩阵部分可用本文件的数字，probe 结果单独补进本节。

## 4. 数据恢复说明

`p2_benchmark` 只在**全部结束**时写 JSON；本次崩溃导致 JSON 未生成，但每个测量点的完整记录都以 `[bench] {...}` JSON 行打印在 `/root/p5_final_matrix.log` 里。本文件的 20 条记录是由这些行逐条恢复、再用同一套定义（`median` 吞吐、`sample std / mean` CV、分位数取 p95 后跨重复取中位数）重新汇总的，脚本与口径与 `nanovllm/benchmarking.py` 一致。

**运维教训（已写入 HANDOFF）**：长矩阵应分点落盘（例如每点写一行 JSONL），不要只在最后写总 JSON；这样崩溃也能保住已完成的数据。
