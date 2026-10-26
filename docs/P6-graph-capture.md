# P6 — MUSA Graph Capture：就是 CUDA Graph 的等价物，但本代栈上被阻塞

> 提问：graph capture 是不是类似 CUDA Graph？能不能用它降低 launch 次数？
> 回答：**是同一个东西**（`torch.musa.MUSAGraph` / `torch.musa.graph` / `torch.musa.graph_pool_handle`，API 与 `torch.cuda.CUDAGraph` 一一对应），仓库里也早就有 capture/replay 骨架；但**实测当前 decode 路径不可捕获**，因此本轮保持 `enforce_eager=True`。

## 1. 仓库里已有的骨架（被 `enforce_eager` 关掉）

- `nanovllm/engine/model_runner.py::capture_cudagraph()`：按 batch 桶 `[1,2,4,8,16,32,…,512]` 逐个捕获 **decode 形状**的前向，`Graph = acc.MUSAGraph`，所有图共享 `graph_pool`，静态输入缓冲 `graph_vars`（input_ids/positions/slot_mapping/context_lens/block_tables/outputs）。
- `run_model()`：`mode=="decode" && !enforce_eager && bs<=512` 时写入 `graph_vars` 后 `graph.replay()`，其余（prefill/mixed、>512）走 eager。
- 但 `api_server.py`、`p2_benchmark.py`、所有 A/B 脚本都硬编码 `enforce_eager=True`，所以这套代码在 S4000 上**从未被验证**。

## 2. 本轮探针结果（`tests/p6_graph_probe.py`，2×S4000，无需加载 30B）

| 检查项 | 结果 | 证据 |
|---|---|---|
| 普通逐元素算子 捕获+重放 | ✅ 数值完全一致 | `plain max_abs=0.000000`（两 rank） |
| **MCCL all-reduce 在图内**（decode 每步 96 次） | ✅ 数值完全一致 | `all_reduce max_abs=0.000000`（两 rank） |
| `torch._grouped_mm`（MoE 专家 GEMM） | ❌ 捕获失败 | `RuntimeError: MUSA error: operation not permitted when stream is capturing`（另有 `FillOp MUDNN failed in: Run`） |
| Triton paged decode kernel | ❌ 捕获失败 | 同一条 `operation not permitted when stream is capturing` |

**结论**：好消息是 **MCCL 集合通信可被捕获**（这是 TP2 图捕获的关键前提），坏消息是 **MoE 的 grouped GEMM 与 Triton kernel 都不可捕获**，而 decode 每一步是 48 ×（attention + MoE），所以整步捕获不可行 → **P6 本轮取消**，`enforce_eager` 保持默认。

> 这是 MUSA runtime 的限制（捕获期间禁止某些 op），不是本项目代码问题。同类报错出现在厂商 MoE 的 MUDNN fill 上，说明不是我们调用姿势不对。

## 3. 顺带完成的有价值改动

- Triton paged decode kernel 的循环上界从 `MAX_CONTEXT_LEN`（constexpr，按上下文长度 pow2 重编译）改为**运行时 `context_len`**：
  - eager 下也不再按上下文桶反复 JIT；
  - 单份编译、单一 grid，是任何未来图捕获的前置条件。
- q·k 点积改为 fp32 累加，与 SDPA 更接近，降低近并列 token 翻转概率（见 `docs/P5-moe-optimization-survey.md` 的 token 一致性数据）。

## 4. 若未来重启 P6：需要什么

1. **可捕获的 MoE**：候选是 `loop` 后端（`index_select` + `index_add_`，未验证是否可捕获；即使可捕获也慢 2.7×，收益需重新评估），或等 MUSA runtime 修复 grouped GEMM 的捕获支持。上游 Triton fused MoE 在本机同样不可捕获。
2. **形状静态的 attention**：当前已通过 runtime-loop paged kernel 解决；SDPA 路径仍有 host 侧 `kv_len` 动态形状（捕获时会冻结为最大值）。
3. **桶与显存**：按实际 `max_num_seqs` 裁剪桶（如 ≤64 → 10 个图），确认 KV 先分配、图池后建，并实测峰值显存增量。
4. **验收门**（已在 `scripts/p5_decide.py --mode graph` 冻结）：token 与 eager 完全一致；b1 吞吐 ≥+15%、b8 ≥+5%；b32/b64 不回归（≥0.98）；CV ≤5%；显存增量 ≤2 GiB。
5. 复现探针：`PYTHONPATH=<repo> python3 tests/p6_graph_probe.py`；期望输出 `[graph-probe] PASS`，当前会以 `FAIL [...]` 退出并在 `results` 里给出具体失败项。

## 5. 结论

- "用 graph 降低 launch 次数"在当前栈上**做不到**，原因是可捕获性而不是设计；本轮的验证避免了把它误当成"打开开关就能快"的低垂果实。
- 已确认 MCCL 可捕获，这为未来（换 MoE 实现或 runtime 升级后）留了明确的入口。
- 默认保持 `enforce_eager=True`，性能数字都在 eager 下测得，报告需注明。
