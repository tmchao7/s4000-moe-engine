# Benchmarks

基线与各阶段优化收益的记录目录。所有数字必须标注：日期、硬件、软件版本、模型、精度、并发参数。

## 文件约定

- `P0-vllm-musa-baseline.md` — vllm-musa（预装 0.29.1）在 2×S4000 上的工具链基线
- `P0-hf-reference.md` — HF transformers 加载 FP8 checkpoint 的正确性参考输出
- `P1-musa-port.md` — 自研引擎 MUSA 移植后 dense 模型基线
- `P2-moe-tp2.md` — 30B-A3B 双卡跑通后的正确性门、吞吐 / TPOT / 显存、loop↔grouped-MM 消融
- `P3-scheduler.md` — FCFS vs stall-free 在线压测（TTFT/TPOT/QPS + mixed step 证据）
- `P3-prefix-placement.md` — 前缀缓存冷/热与专家放置（linear/round_robin/profiled）对比
- `P4-serving.md` — OpenAI 兼容接口的真实 30B 开放环压测
- `data/` — 原始 JSON/JSONL 与渲染出的 PNG（不入库）

## 统一测量方法

- **黄金规则**：任何数字只有在正确性门通过后才有意义。30B 的旧 24.4 tok/s 来自乱码输出，已作废；当前基线统一用 `loop` 后端 + 冻结的参考输出。
- **正确性**：`temperature=0` 贪心；tiny 模型四路（TP1/TP2 × loop/grouped）逐 token 一致；30B 固定 4 条 prompt 输出连贯且**同一后端稳态可复现**（详见 `P2-moe-tp2.md`）。
- **离线吞吐**：随机 token prompt（默认 128 in / 32 out），每点先 warmup 1 次再测 3 次，报告吞吐中位数、变异系数（CV）、TTFT p95 与 TPOT p95 的中位数。
- **在线压测**：开放环客户端按目标 QPS × 负载比例注入请求，报告达成 QPS、成功率、TTFT p50/p95、E2E p95，并从 `/metrics` 抓取该负载窗内的 `engine_mixed_steps_total` 等引擎侧计数。
- **并发点**：离线 1/2/4/8/16/32/64；在线 `--requests 32`（Little's law 下 100% 负载点即 batch≈32）。
- **已知偏差**：本轮吞吐测量用 `temperature=0`（贪心），与生产常见 temperature=0.6 不同；温度只影响采样，不影响前向开销，后续如需与外部数字对比应统一标注。
