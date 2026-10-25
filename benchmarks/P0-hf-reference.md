# P0：HF transformers 加载 Qwen3-30B-A3B-FP8（S4000 / MUSA）

- **日期**：2026-10-25（状态：加载中，结果 JSON 落盘于服务器 `/root/p0_hf_reference.json`，脚本 `scripts/p0_hf_fp8_check.py`）
- **环境**：transformers 5.13.0 + torch_musa 2.11，`device_map="auto"` 双卡切分

## 过程与发现

1. **transformers 明确不支持在 S4000 上做 FP8 计算**，日志原文：`FP8 quantized models is only supported on GPUs with compute capability >= 8.9 (e.g 4090/H100), actual = 2.2. We will default to dequantizing` —— 官方降级路径就是**反量化**，验证了本项目 P2 "FP8 → bf16 反量化加载"的设计方向。
2. **torch_musa 原生实现了 `torch._grouped_mm`**（transformers 的 MoE experts 走 grouped GEMM 路径并真正调用了它）—— P2 的 MoE 实现可以直接用分组 GEMM，无需手写 expert loop。重要利好。
3. 遇到并修复的问题（已 patch 进脚本，都是 MUSA 适配的真实素材）：
   - accelerate 拆分后 MoE `offs` 张量设备不一致（musa:0 vs musa:1）→ monkeypatch 对齐到权重设备
   - transformers ≥5 的 `apply_chat_template` 默认返回 BatchEncoding 而非 tensor
   - muDNN 报 `NOT_SUPPORTED in Tensor::SetType`（FP8 dtype 相关，非致命）
4. **CPU 反量化路径极慢**：~54G 专家权重在 CPU 上反量化，480% CPU 持续 >30 分钟未完成 → 官方栈的加载体验不可用级别。**这正是 P2 自研高效反量化加载器（GPU 上分块反量化 / 流式加载）的直接动机。**

## 结论（暂定，待加载完成后补充）

- 正确性参考输出的价值不变：等反量化完成后跑 3 条固定 prompt 的贪心输出，作为 P2 自研引擎的逐 token 对齐目标。
- 对 P2 的输入：反量化逻辑可直接参考 transformers 的 dequant 实现（e4m3 + block scale），但要用 GPU 化 / 流式方式重写。
