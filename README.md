<p align="center">
<img width="220" src="assets/logo.png">
</p>

# moe-server-s4000

基于 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的国产 GPU（摩尔线程 MTT S4000 / MUSA）MoE 大模型双卡推理引擎。

> 本仓库 fork 自 nano-vllm（MIT License, © 2025 Xingkai Yu），上游代码保留完整 git 历史，本项目所有改动可通过 `git diff upstream/main` 查看。上游原文档见 [docs/upstream-readme.md](docs/upstream-readme.md)。

## 项目定位

在 ~1,400 行的轻量推理引擎 nano-vllm 基础上（上游已含 continuous batching、prefix caching、chunked prefill、CUDA graph、TP），针对摩尔线程 S4000 做三件事：

1. **MUSA 适配**：把引擎从 CUDA 生态移植到 torch_musa + MCCL + Triton-mtgpu；
2. **MoE 模型支持**：新增 Qwen3-MoE（30B-A3B）模型实现，含 FP8 量化权重反量化加载；
3. **MoE 加速与调度**：packed expert + MUSA grouped GEMM、TP2+EP2、stall-free mixed prefill/decode；
4. **可观测服务**：请求级 TTFT/TPOT 指标、可复现 benchmark、OpenAI 兼容流式 API。

目标硬件与软件栈：

| 项 | 配置 |
|---|---|
| GPU | 2× MTT S4000（48GB / 卡，PCIe Gen5 x16） |
| 软件栈 | torch 2.11 + torch_musa 2.11、MCCL 2.11、Triton（mtgpu backend）、Ubuntu 22.04 |
| 目标模型 | Qwen/Qwen3-30B-A3B-Instruct-2507-FP8（30.5B 总参 / 3.3B 激活，128 专家 top-8） |

## 架构

```
├── nanovllm/            # 引擎本体（fork 自上游）
│   ├── engine/          # LLMEngine / Scheduler / ModelRunner / BlockManager
│   ├── models/          # qwen3.py（上游）→ + qwen3_moe.py（本项目）
│   ├── layers/          # attention / linear / layernorm / rope / sampler
│   └── ...
├── docs/                # 交接 / 计划 / 问题台账 / 架构（活文档仅 7 个,见 AGENTS.md）
├── benchmarks/          # 当前轮报告;历史轮次在 benchmarks/archive/
└── scripts/             # 压测 / 基准 / 决策脚本;一次性实验脚本在 scripts/archive/
```

Agent/新人入口见 [AGENTS.md](AGENTS.md)(常用命令、红线、SOP);详细设计见 [docs/architecture.md](docs/architecture.md);当前迭代计划见 [docs/PLAN.md](docs/PLAN.md)。

## 当前进度(截至 P14 轮,全部判决完毕;详见 [docs/HANDOFF.md](docs/HANDOFF.md) 与 [docs/CLOSEOUT.md](docs/CLOSEOUT.md))

| 阶段 | 状态 | 关键数字 |
|---|---|---|
| P0–P1 基线与 MUSA 移植 | 完成 | vllm-musa 8.4 tok/s → 自研引擎单卡 113.5 tok/s(b64);首 token 与 HF 100% 对齐 |
| P2–P2e MoE 双卡 + 优化 | 完成 | 30B TP2+EP2 packed experts + grouped_mm 比 loop 快 3.0–3.2×;离线 b32 133 tok/s、b64 141 tok/s |
| P3–P4 调度与服务化 | 完成 | stall-free 混步调度;OpenAI 兼容 SSE API;短 prompt 四档 Poisson ×3 全 100% SLO(std+strict) |
| P5–P6 指标/后端/长上下文 | 完成 | 指标 v3(双 SLO/Wilson CI);**triton_paged 按五层门转正**(TPOT p95 −12~18%);I8 长上下文重测 + 调度三缺口修复 |
| P7–P9 优化探索 | 完成 | 投机解码 NO-GO(α≈0);I19 因果链否定(~20–30 ms/步未解释固定等待);整图否决(I29,TP2 replay 楔死);chunk-2048 把 1k TTFT 砍 −69~73%(50% 档 1.036 s、SLO 100%) |
| P10–P12 分段图与在线判定 | 完成 | 分段图(图内零 MCCL)离线 30B b1 **−23.7%**、768 图/rank 零失败、跨臂逐位 0 翻转;**在线 A/B 判定 FAIL**(并发 14–30 撞桶 padding 税,慢 5.9–14.7%,同 session 复跑确认)→ 不转正、默认维持 eager |
| P13 硬骨头攻坚 | 完成 | **I30 关闭**(同 session ABA:现树快于 pre-P8 栈 23–51%);**I33 修复验证并关闭分支**(密桶 1056 图零失败 + pad 回退,padding 税证实并消除 0.9 档 +14.7%→+2.3%,残余 AR 串行化 +4.7% 超门 → 段图定格单流开关);**I8 闭合**(冷缓存塌=chunk 粒度);**容量下界 1.070 req/s** |
| P14 收官补测 | 完成 | `--prompt-seed-per-repeat` 根治暖缓存伪象;I12 exit 看门狗;**容量拐点 1.6–1.8×、硬饱和 ≈1.2 req/s**;1k 冷态 1.0 档 chunk-2048 过 TTFT 门(1.08–1.12 s);I29 官方线索(MCCL 混用,缓解非修复) |

正确性门:tiny TP1/TP2 × loop/grouped/bmm 六配置逐 token 全等;跨后端比较采用"候选确定性 + 五层等价门"(数值契约见 [docs/numerics-policy.md](docs/numerics-policy.md))。历史轮次报告(P0–P8)在 [benchmarks/archive/](benchmarks/archive/),当前轮在 [benchmarks/](benchmarks/)。

## 结果一览(图表由 `scripts/render_metrics.py` 从 `benchmarks/data/` 证据 JSON 程序化生成)

### 端到端与离线吞吐

<p align="center">
<img width="420" src="assets/charts/baseline_throughput.png">
<img width="520" src="assets/charts/offline_moe_throughput.png">
</p>

### 在线服务能力(30B 生产栈)

容量拐点 1.6–1.8×,硬饱和 ≈1.2 req/s,1.0× 档双口径 SLO 100%:

<p align="center">
<img width="620" src="assets/charts/capacity_curve.png">
</p>

分段图的完整闭环——离线单流 −23.7%,在线被桶 padding 税否决,密桶+回退修复后 padding 税消除、残余为桶边界 AR 串行化,最终按预注册门关闭:

<p align="center">
<img width="620" src="assets/charts/tpot_tiers.png">
<img width="520" src="assets/charts/segstep.png">
</p>

### 1k 长输入:冷缓存塌方 = chunk 粒度(I8 终局)

<p align="center">
<img width="480" src="assets/charts/longctx_ttft.png">
</p>

## 快速开始

```bash
pip install -e .
python example.py
```

MUSA 环境各阶段运行方式见 [AGENTS.md](AGENTS.md) 常用命令节。

双卡正确性回归(按顺序,任何一步失败都不要往下跑性能;均在服务器跑,tiny 模型在 `/tmp/tiny_moe`):

```bash
PYTHONPATH=. python3 tests/p2_grouped_mm.py       # TP1/TP2 × loop/grouped/fused + round_robin + 重复确定性(六配置全等 rc=0)
PYTHONPATH=. python3 tests/p6_bmm_dense_tiny.py   # bmm_dense decode 后端六配置全等
python3 -m pytest tests/test_benchmarking_v3.py   # 压测客户端单测 15/15
PYTHONPATH=. python3 tests/p8_scheduler_gaps.py   # 调度缺口单测 6/6
```

消融 benchmark(模型只加载一次,进程内切换后端;先跑正确性门再测性能):

```bash
PYTHONPATH=. python3 scripts/p2_benchmark.py \
  --batches 1,2,4,8,16,32,64 --prompt-len 128 --output-len 32 --repeats 3 \
  --moe-backends loop,grouped_mm --scheduler-policy fcfs \
  --verify-prompts --reference-outputs benchmarks/p2-reference-8tok.json \
  --output benchmarks/data/p2_ablation_full.json
```

OpenAI 兼容服务与开放环压测：

```bash
pip install -e '.[serve]'
# 生产栈必须显式带 --decode-moe-backend bmm_dense:该参数默认跟随 --moe-backend(grouped_mm),
# 漏掉则 decode 跑在 grouped_mm 上,不是本项目的生产配置(见 docs/ISSUES.md I32)。
PYTHONPATH=. python3 -m nanovllm.entrypoints.openai.api_server \
  --model /root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --served-model-name qwen3-30b-a3b --scheduler-policy stall_free \
  --decode-moe-backend bmm_dense

PYTHONPATH=. python3 scripts/benchmark_online.py --model qwen3-30b-a3b \
  --max-qps 0.93 --requests 32 --max-tokens 64 --fractions 0.25,0.5,0.75,1.0 \
  --output benchmarks/data/online.json
```

## License

MIT（沿用上游）
