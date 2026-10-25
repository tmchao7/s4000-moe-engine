# Benchmarks

基线与各阶段优化收益的记录目录。所有数字必须标注：日期、硬件、软件版本、模型、精度、并发参数。

## 文件约定

- `P0-vllm-musa-baseline.md` — vllm-musa（预装 0.29.1）在 2×S4000 上的工具链基线
- `P0-hf-reference.md` — HF transformers 加载 FP8 checkpoint 的正确性参考输出
- `P1-musa-port.md` — 自研引擎 MUSA 移植后 dense 模型基线
- `P2-moe-tp2.md` — 30B-A3B 双卡跑通后的吞吐 / 显存 / 通信数据
- `P3-scheduler-*.md` — 调度优化逐项收益

## 统一测量方法

- 输入/输出长度：随机 100–1024 tokens（沿用上游 bench.py 方法），另加固定 prompt 集做正确性回归
- 指标：吞吐（tokens/s）、TTFT（ms）、TPOT（ms）、显存峰值（MB）
- 并发点：1 / 8 / 32 / 128 请求
- 解码参数：吞吐测试 temperature=0.6；正确性测试 temperature=0（贪心）
