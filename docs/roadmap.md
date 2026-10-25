# Roadmap

> 每阶段有明确验收标准；时间按课余 ~10h/周估算，合计约 10–12 周。

## P0 基线（1 周）—— 当前阶段

**目标**：建立性能与正确性的参照系。

- [ ] 下载 `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`（~31.2G）到数据盘 `/root/autodl-tmp`
- [ ] 下载 `Qwen/Qwen3-8B`（dense, bf16, ~16G）到数据盘
- [ ] vllm-musa（预装 0.29.1）TP2 跑 Qwen3-8B：不同并发下吞吐 / TTFT / TPOT → `benchmarks/P0-vllm-musa-baseline.md`
- [ ] HF transformers 加载 FP8 checkpoint（自动反量化）跑贪心输出，存固定 prompt 集的参考输出 → `benchmarks/P0-hf-reference.md`
- [ ] 实测 vllm-musa 能否加载 FP8 checkpoint（预期失败，失败原因记录，作为自研动机素材）
- [ ] 复核环境结论：Triton-mtgpu / MCCL / SDPA（已完成初验，写入 benchmarks）

## P1 移植 MUSA（2 周）

**目标**：nano-vllm 单卡在 S4000 上跑通 Qwen3 dense。

- [ ] 设备抽象：`cuda` → `musa`（model_runner / utils context）
- [ ] Triton paged attention 内核 → SDPA 替换（P1 简化实现，正确性优先）
- [ ] CUDA graph：先 enforce_eager；torch_musa graph capture 实测后决定是否启用
- [ ] MCCL 分布式初始化替换 NCCL
- [ ] **验收**：Qwen3-8B（或 4B）单卡贪心输出与 HF 逐 token 对齐；吞吐记录

## P2 Qwen3-MoE + FP8 加载 + 双卡 TP（3 周）

**目标**：30B-A3B 在 2×S4000 上跑通。

- [ ] `models/qwen3_moe.py`：router（sigmoid + top-8）、expert MLP、权重映射
- [ ] FP8（e4m3 + block scale）→ bf16 反量化加载器，与 transformers 数值对齐
- [ ] MoE 层 TP 切分：128 专家按 rank 对半切（expert-parallel in TP），dense 部分列/行切分
- [ ] **验收**：FP8 checkpoint 加载后贪心输出对齐 HF 参考；吞吐 vs 显存占用曲线；每层 allreduce 通信量实测

## P3 调度层扩展（3–4 周，简历核心）

**目标**：在调度层做出可量化的优化。

- [ ] overlap scheduler：CPU 调度与 GPU 前向重叠（参考 Sarathi-Serve / mini-sglang）
- [ ] radix 前缀缓存：上游 block-hash 前缀缓存升级为 radix tree，多轮对话场景命中率与 TTFT 对比
- [ ] MoE 感知调度：专家负载感知的 batch 组装（探索性，视 P2 profiling 结果定方向）
- [ ] **验收**：每项优化独立 benchmark（吞吐 / TTFT / TPOT 曲线），写入 `benchmarks/`

## P4 服务化与报告（1–2 周）

- [ ] OpenAI 兼容 API（FastAPI，流式）
- [ ] 完整压测报告 + README 定稿（架构图、性能曲线、MUSA 适配踩坑记录）

## 长期打磨（不设期限）

- Triton-mtgpu paged attention / fused MoE kernel
- W4/AWQ 权重加载（GLM-4.5-Air 106B stretch goal）
- torch_musa graph capture / torch.compile
- 投递目标：2026 实习（推理引擎 / infra 方向）

## 风险清单

| 风险 | 缓解 |
|---|---|
| 数据盘 50G 且不可扩容 | 采用 FP8 checkpoint（31.2G）；两模型分时下载，用完可删 dense 基线模型 |
| torch_musa 算子缺失 | 逐个 fallback 到等价组合；记录到 docs 作为适配成本素材 |
| SDPA 起步性能低于 vllm-musa | 属预期；P3+ 用 Triton-mtgpu 内核追赶，差距即优化空间 |
| vllm-musa 对 Qwen3-MoE 支持不全 | P0 实测；基线回退到 Qwen3-8B dense + HF 正确性对齐 |
| MoE TP 通信成为瓶颈 | PCIe Gen5 x16 带宽高，预估占比 <10%；P2 实测验证 |
