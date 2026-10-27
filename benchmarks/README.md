# Benchmarks

基线与各阶段优化收益的记录目录。所有数字必须标注：日期、硬件、软件版本、模型、精度、并发参数。

## 文件约定

- `P0-vllm-musa-baseline.md` — vllm-musa（预装 0.29.1）在 2×S4000 上的工具链基线
- `P0-hf-reference.md` — HF transformers 加载 FP8 checkpoint 的正确性参考输出
- `P1-musa-port.md` — 自研引擎 MUSA 移植后 dense 模型基线
- `P2-moe-tp2.md` — 30B-A3B 双卡跑通后的正确性门、吞吐 / legacy ITL / 显存、loop↔grouped-MM 消融
- `P3-scheduler.md` — FCFS vs stall-free 在线机制压测（TTFT/E2E/QPS + mixed step 证据）
- `P3-prefix-placement.md` — 前缀缓存冷/热与专家放置（linear/round_robin/profiled）对比
- `P4-serving.md` — OpenAI 兼容接口的真实 30B 开放环压测
- `P5-profiling.md` — schema v2 指标校准、真实 30B 分段 profiling 与证据驱动的优化路线
- `p5-profile-decision.json` — 可审计的 profiling 阈值、观测值和自动决策（原始大 JSON 留在 `data/`）
- `P5-attention-ab.md` — SDPA vs Triton paged decode 的同进程 A/B（正确性 + 长上下文性能）
- `p5-attention-ab-decision.json` — 冻结的 attention 采纳门（token 全等 / TPOT ≤0.90 / 吞吐 ≥1.00 / CV ≤5%）
- `P5-final-matrix.md` — schema-v2 终局矩阵 + 前缀缓存冷/热探针（第 4 发加载）
- `P5-serving-v2.md` — OpenAI 接口的 Poisson 开放环容量/goodput（第 5 发加载）
- `P6-serving-v3.md` — decode=bmm_dense 后的在线容量复测（与 v2 同表，四档全 100% SLO）
- `P6-nightly.md` — P6 夜间迭代总表（bmm_dense 采用 / b32-b64 治好 / attention keep_sdpa / 图控制组 0/5 与 I4 定位）
- `P6-i5-i6-i7.md` — P6 覆盖补测：b64 采用栈点（141.5 tok/s）、前缀缓存冷热（TTFT -25.6%）、SLO 软边界修正
- `p5-moe-ab-decision.json` — MoE 后端的采纳门（token 全等 / 吞吐 ≥1.08 / TPOT ≤1.00 / CV ≤5%）
- `data/` — 原始 JSON/JSONL 与渲染出的 PNG（不入库）

## 相关分析文档（非 benchmark 数字）

- `../docs/P5-metrics-audit.md` — 指标口径与工业界（vLLM / GenAI-Perf / MLPerf / Databricks）的逐项对照与偏差
- `../docs/engine-vs-vllm-sglang.md` — 本引擎相对 vLLM/SGLang 的能力矩阵与刻意不做项
- `../docs/P5-moe-optimization-survey.md` — MoonEP/UniEP/MegaMoE/上游 fused MoE 的调研与实测（含为什么没换默认）
- `../docs/P6-graph-capture.md` — MUSA graph capture 的可行性结论（当前不可捕获）

## 统一测量方法

- **黄金规则**：任何数字只有在正确性门通过后才有意义。30B 的旧 24.4 tok/s 来自乱码输出，已作废；当前基线统一用 `loop` 后端 + 冻结的参考输出。
- **正确性**：`temperature=0` 贪心；tiny 模型四路（TP1/TP2 × loop/grouped）逐 token 一致；30B 固定 4 条 prompt 输出连贯且**同一后端稳态可复现**（详见 `P2-moe-tp2.md`）。
- **离线吞吐 v2**：随机 token prompt，每点 warmup 1 次、正式 3 次；CV=`sample std / mean`，分别报告请求级 TPOT 与 token-gap ITL。P2 冻结数据属于 legacy v1，字段名不回写。
- **在线压测 v2**：Poisson/固定开放环注入；只有收到 `[DONE]` 和真实 token usage 才成功，报告 request/input/output/total throughput、TTFT/TPOT/ITL/E2E、goodput 与最大并发。
- **样本量**：机制测试保留原 32 请求/点；P5 容量结论至少 128 请求/点，最终 75%/90% 点使用 256 请求。
- **已知偏差**：本轮吞吐测量用 `temperature=0`（贪心），与生产常见 temperature=0.6 不同；温度只影响采样，不影响前向开销，后续如需与外部数字对比应统一标注。
- `P7-phase0.md` — 投机解码 Phase 0 实测与 go/no-go（**NO-GO**：α=0.006、c=0.384、0.73×）+ 每步固定成本发现
