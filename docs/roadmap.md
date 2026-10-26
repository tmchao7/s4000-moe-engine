# Roadmap

> 每阶段有明确验收标准；时间按课余 ~10h/周估算，合计约 10–12 周。

## P0 基线（已完成）

**目标**：建立性能与正确性的参照系。

- [x] 下载 Qwen3-30B-A3B FP8 与 Qwen3-8B。
- [x] 建立 vllm-musa Qwen3-8B TP2 基线与 HF 正确性参照。
- [x] 验证 Triton-mtgpu、MCCL、MUSA SDPA 和 `torch._grouped_mm`。

## P1 移植 MUSA（已完成）

**目标**：nano-vllm 单卡在 S4000 上跑通 Qwen3 dense。

- [x] MUSA/MCCL 设备抽象、SDPA attention、MUSAGraph 兼容层。
- [x] Qwen3-8B 贪心与 HF 对齐；batch 64 达到 113.5 tok/s。

## P2 Qwen3-MoE + FP8 加载 + 双卡 TP/EP（已完成）

**目标**：30B-A3B 在 2×S4000 上跑通。

- [x] softmax top-8 router、FP8 block dequant、Attention TP2 + experts EP2。
- [x] 修复 loader 静默漏权重、RMSNorm 原位污染和 decode KV 尾部非确定性。
- [x] packed experts + MUSA grouped GEMM；TP1/TP2、loop/grouped-MM 四路一致。
- [x] 30B 四组中英文 prompt 输出连贯。
- [x] 完成 loop/grouped-MM batch 1–64 三次重复性能消融：grouped-MM **74.0 tok/s @b32 / 120.4 tok/s @b64**，比 loop 快 **3.0–3.2×**，CV ≤1.1%，验收门 PASS。
- [x] 修复 grouped-MM 用原子 `index_add_` 合并 top-k 导致的贪心不可复现：改为 slot scatter + 固定顺序求和；30B 上两端确定性、loop↔grouped 4/4 prompt 逐 token 一致。

详见 [benchmarks/P2-moe-tp2.md](../benchmarks/P2-moe-tp2.md)。

## P3 调度层扩展（已完成）

**目标**：在调度层做出可量化的优化。

- [x] decode-first stall-free scheduler：同一步混合 decode 与 bounded prefill chunks。
- [x] block-hash prefix cache 上的 longest-prefix admission + starvation guard。
- [x] 逐层 expert load 采集与静态 profile-guided placement 生成器。
- [x] 完成 FCFS/stall-free 的在线独立 benchmark（TTFT/TPOT/达成 QPS + mixed step 证据），见 [benchmarks/P3-scheduler.md](../benchmarks/P3-scheduler.md)。

## P4 服务化与报告（已完成）

- [x] OpenAI chat completions API（FastAPI，SSE 流式、取消、metrics）。
- [x] 可复现 benchmark JSON、Markdown 与 PNG 生成脚本。
- [x] 修复 transformers 5.x `apply_chat_template` 返回 `BatchEncoding` 导致 API 无法接真实 tokenizer 的问题（并对该分支补了契约测试）。
- [x] 真实 30B API batch 32 开放环压测与最终报告数字，见 [benchmarks/P4-serving.md](../benchmarks/P4-serving.md)。

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
