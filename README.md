<div align="center">
<img src="https://capsule-render.vercel.app/api?type=soft&height=150&color=0:1E3A8A,55:2563EB,100:0D9488&text=s4000-moe-engine&fontSize=42&fontColor=ffffff&desc=Qwen3-30B-A3B%20MoE%20%C2%B7%20MUSA%20%2F%20MTT%20S4000%20%C3%972%20%C2%B7%20TP2%2BEP2%20%C2%B7%20pre-registered%20frozen-gate%20A%2FB&descSize=15&descAlignY=76">
</div>

<div align="center">

![License](https://img.shields.io/badge/License-MIT-blue)
![GPU](https://img.shields.io/badge/GPU-MTT%20S4000%20%C3%972-EA580C)
![Model](https://img.shields.io/badge/Model-Qwen3-30B-A3B-2563EB)
![Parallel](https://img.shields.io/badge/Parallel-TP2%20%2B%20EP2-0D9488)
![Offline](https://img.shields.io/badge/Offline-141.5%20tok%2Fs%20%28b64%29-1B9E77)
![Online](https://img.shields.io/badge/Online-%E2%89%A51.07%20req%2Fs%20%C2%B7%20100%25%20SLO-7C3AED)
![Repro](https://img.shields.io/badge/repro-6%E9%85%8D%E7%BD%AE%20rc%3D0-111827)

</div>

跑在摩尔线程 MTT S4000（MUSA 全栈，无 FP8 算力）上的 Qwen3-30B-A3B 双卡 MoE 推理引擎。TP2/EP2、自研 Triton paged attention、stall-free 混批调度、OpenAI 兼容流式服务。引擎底座是 MIT 协议的 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)（约 1.4k 行）；MUSA 适配、MoE 模型、调度/服务化和全部平台能力是本项目写的，上游版权见 [LICENSE](LICENSE)。

所有性能结论出自预注册冻结门的 A/B 实验：先写判据再跑，跑完不改阈值，负结果照录——P0 到 P26 共 27 轮，判决速览见 [docs/HANDOFF.md](docs/HANDOFF.md) §1（PLAN 保留冻结门与 P16–P18 判决）；实验轮的报告与证据 JSON+sha256 在 [benchmarks/](benchmarks/) 及 [benchmarks/archive/](benchmarks/archive/)（P25 评审报告在 docs/P25-review.md），纯整理轮以台账判决为准。

## 结果

图表由 [`scripts/render_metrics.py`](scripts/render_metrics.py) 从 `benchmarks/data/` 的证据 JSON 生成，一条命令可重渲；每个数字都能对到对照臂、冻结门和存档文件。注意：**证据 JSON 不入库**（`benchmarks/data/` 在 .gitignore，边界说明见 [benchmarks/README.md](benchmarks/README.md)）——新 clone 上重渲需先自产 measurements；对页面上数字以各轮报告 md+sha256 为准。徽章 141.5 tok/s (b64) 与下方 `offline_moe_throughput.png` 的 b64 ≈120 tok/s **口径不同**：前者出自 [benchmarks/archive/P6-nightly.md](benchmarks/archive/P6-nightly.md)（b64、128-in/128-out、I5 栈），后者出自 p2 消融（b64、128-in/32-out、fcfs），同为存档实测、非同一负载，引用时注明口径。

| 端到端加速（8B 基线） | 30B 离线吞吐 × MoE 后端 |
|---|---|
| ![基线对比](assets/charts/baseline_throughput.png) | ![离线吞吐](assets/charts/offline_moe_throughput.png) |

在线容量曲线（拐点 1.6–1.8×，硬饱和 ≈1.2 req/s，1.0× 档双口径 SLO 100%）：

<p align="center">
<img width="640" src="assets/charts/capacity_curve.png">
</p>

分段图走完了一个完整的判决闭环：离线单流 −23.7% 成立，在线 A/B 被桶 padding 税否决（慢 5.9–14.7%），密桶+回退把 padding 税消除后仍剩桶边界 AR 串行化——最终按预注册门关闭，只留单流开关。

| 在线 TPOT p95 分档 | 30B 离线 decode 步时 |
|---|---|
| ![TPOT 分档](assets/charts/tpot_tiers.png) | ![分段图步时](assets/charts/segstep.png) |

1k 长输入的冷缓存塌方，根因是 chunk 粒度（chunk-2048 冷态 TTFT p50 1.08–1.12 s，过 ≤2 s 门）：

<p align="center">
<img width="500" src="assets/charts/longctx_ttft.png">
</p>

**口径与可信度边界**（照实写）：负载与压测客户端是自建的（客户端有 28 个单测、schema 版本化），判据是自己冻结的，证据来自单一环境（2×S4000 租用实例）。冻结门解决的是"自己骗自己"，替代不了第三方验证——标准负载（ShareGPT 类 trace）交叉对账是已知未做项。对比基线 = vllm-musa 0.29.1（2026-10-25 同栈同卡实测）；vLLM 迭代很快，引用加速比请连同基线版本与日期一起引。

## 复现

| 项 | 配置 |
|---|---|
| GPU | 2× MTT S4000（48GB / 卡，PCIe Gen5 x16） |
| 软件栈 | torch 2.11.0.post1 + torch_musa 2.11.0.post1、MCCL 2.11、Triton（mtgpu backend）、Ubuntu 22.04 |
| 模型 | Qwen/Qwen3-30B-A3B-Instruct-2507-FP8（30.5B 总参 / 3.3B 激活，128 专家 top-8）；tiny 模型由 `scripts/make_tiny_moe.py` 生成 |
| 环境搭建 | [docs/SETUP.md](docs/SETUP.md)——软件栈、模型下载、复现链路（本仓库仅支持 MUSA 后端） |

```bash
pip install -e .
NANOVLLM_MODEL=/data/models/Qwen3-0.6B python example.py   # 单卡冒烟
```

双卡正确性回归（顺序执行，任何一步失败都不要往下跑；tiny 模型在 `/tmp/tiny_moe`）：

```bash
PYTHONPATH=. python3 tests/p2_grouped_mm.py       # TP1/TP2 × loop/grouped/fused + round_robin + 重复确定性,六配置全等 rc=0
PYTHONPATH=. python3 tests/p6_bmm_dense_tiny.py   # bmm_dense decode 后端六配置全等
python3 -m pytest tests/test_benchmarking_v3.py   # 压测客户端单测,28 passed
PYTHONPATH=. python3 tests/p8_scheduler_gaps.py   # 调度缺口单测 6/6
```

消融 benchmark（模型只加载一次，进程内切换后端；先过正确性门再测性能）：

```bash
PYTHONPATH=. python3 scripts/p2_benchmark.py \
  --model /data/models/Qwen3-30B-A3B-Instruct-2507-FP8 \
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
  --model /data/models/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --served-model-name qwen3-30b-a3b --scheduler-policy stall_free \
  --decode-moe-backend bmm_dense

PYTHONPATH=. python3 scripts/benchmark_online.py --model qwen3-30b-a3b \
  --max-qps 0.93 --requests 32 --max-tokens 64 --fractions 0.25,0.5,0.75,1.0 \
  --output benchmarks/data/online.json
```

图表重渲（改了 benchmark 数据之后跑一次，PNG 与 README 数字保持同源；`benchmarks/data/` 不入库，新 clone 需先跑 benchmark 生成数据）：

```bash
python3 scripts/render_metrics.py    # 需要 matplotlib + SciencePlots
```

## 已知限制与负结果

照实写。每一项都有判决报告，细节见 [docs/CLOSEOUT.md](docs/CLOSEOUT.md) 与 [docs/ISSUES.md](docs/ISSUES.md)。

| 项 | 结论 |
|---|---|
| MUSA Graph replay | TP2 引擎级间歇楔死（I29，驱动层，无诊断工具可见）→ 生产默认 eager；MCCL 混用开关只缓解不修复 |
| KV int8 量化 | 三种方案机制逐位正确、质量门连续 FAIL——Qwen3 K/V 通道内动态范围 ~500× 超出 int8 的 127 级网格（I20 终判）；fp8 e4m3 在 MUSA 上是静默 no-op |
| 投机解码 | 实测 α≈0 且 c=0.39，解析式外推与实测误差 0.3%，NO-GO |
| 分段图 | 离线 b1 −23.7% 但在线 A/B FAIL（padding 税+AR 串行化），不转正，定格单流开关 |
| 厂商算子 | `torch._grouped_mm` 不可捕获、mp22 fast path 数值错误、launch watchdog 不可配——各自有绕过方案 |

## 目录

```
├── nanovllm/            # 引擎本体(Python 包名沿用底座)
│   ├── engine/          # LLMEngine / Scheduler / ModelRunner / BlockManager
│   ├── models/          # qwen3_moe.py(本项目,Qwen3-MoE TP2/EP2)
│   ├── layers/          # attention(自研 Triton paged) / linear / layernorm / rope / sampler
│   └── ...
├── docs/                # 交接 / 计划 / 问题台账 / 架构 / 对标
├── benchmarks/          # 当前轮报告;历史轮次在 benchmarks/archive/
└── scripts/             # 压测 / 基准 / 决策 / 图表脚本;一次性实验脚本在 scripts/archive/
```

详细设计见 [docs/architecture.md](docs/architecture.md)；当前迭代计划见 [docs/PLAN.md](docs/PLAN.md)；与 vLLM/SGLang 的能力对标见 [docs/engine-vs-vllm-sglang.md](docs/engine-vs-vllm-sglang.md)。

<details>
<summary><b>P0–P26 判决史（点击展开）</b></summary>

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
| P15 autotune + PD 分离 | 完成 | E1 kernel autotune 内核 **1.59–2.90×**(数值门 FAIL→默认不翻,opt-in);I29 新栈维持(init 楔+驱动级残留);**PD 分离单机双卡全闭环**:M1/M2 bitwise 交接 + **在线 A/B ITL p95 −16~25%、client e2e −41~48%**(吞吐换延迟,实测验证);新发现 I34(token 依赖引擎状态历史) |
| P16–P18 隔夜轮(调度做足/KV 量化/PD) | 完成 | **统一混合批**混合步 −25%/−15%(opt-in);**KV LRU 淘汰**(两级 victim,默认开);**优先级调度**高优 TTFT **42.7×**;KV int8 容量 2.0×/数值方案待换;**PD Poisson 双档 POSITIVE**(0.79×/0.82×);30B E1 负结果闭合 |
| P19–P21 收敛与整理 | 完成 | **I20 终判**(int8 三方案机制全对/质量门连续 FAIL——本模型 K/V 通道内动态范围 ~500× 超 int8 网格);P16-A 30B 证据(0.977+0 翻转);PD 前缀复用 **5.7×** + priority×PD 全链路 + PD OpenAI 端点;整理轮(目录树去 AI 味/过期文档清理/CLOSEOUT 遗留问题总账) |
| P22 采样面对齐(vLLM/SGLang 服务层) | 完成 | **top_p/top_k/min_p + presence/frequency/repetition penalty + 引擎级 stop_token_ids + logprobs(top-k) + /v1/completions(流式+非流式)**;默认参数走逐字 legacy 路径(改造前基线 bitwise 全等);top_p 激活 TPOT +9.1%;API 冒烟 6/6 |
| P23 overlap scheduler | 完成 | vLLM V1/SGLang 单飞流水线(opt-in):**稳态输出与串行逐位全等**(greedy+采样,30B 同);错峰翻转 0.347%;抢占场景 12/12 完成;I36 投机调度块边界 off-by-one 自查自修(MUSA 提交卡死 → launch 时刻块覆盖检查);b1/b16 步时 0.997× |
| P24 结构化输出(xgrammar) | 完成 | **guided_json/guided_regex/response_format json_schema**;位掩码与官方 CPU apply 逐位一致;四 schema 16/16 合法 + grammar 终止;约束开销 **1.028×**;API 端到端冒烟 |
| P25 评审与修复 | 完成 | 深度代码审查 + 2026-11 竞品比对;两个已复现的 overlap 状态机缺陷当日修复(I37/I38,排空式收尾 bitwise 全等);`--scheduler-overlap` CLI 补齐;整理轮归档 18 个文件、引擎 -357 行 |
| P26 收官轮 | 完成 | 测试映射矩阵 + 一键回归(15 套,含冻结门守卫)+ 文档一致性 linter + 公开环境文档;口径与可信度边界入 README;悬置项终判 + 停止线;收官验证 15/15 |
| P27 验收修复轮 | 完成 | 独立验收审计的 high/medium 全修:overlap stale prefill 行 + 冻结记账(I39/I40)、p24 greedy + informational g5、pytest 干净 checkout、公开复现链 `--model`/extras、16 套回归(新增 CPU 边界门);强门 **16/16** |

正确性门:tiny TP1/TP2 × loop/grouped/bmm 六配置逐 token 全等;跨后端比较采用"候选确定性 + 五层等价门"(数值契约见 [docs/numerics-policy.md](docs/numerics-policy.md))。历史轮次报告(P0–P17/P19 有独立报告)在 [benchmarks/archive/](benchmarks/archive/);P18/P20/P21 为并入相邻报告/台账的测量轮,不单列报告。当前轮(P22 起)在 [benchmarks/](benchmarks/)。

</details>

## License

MIT（底座沿用上游，版权声明见 [LICENSE](LICENSE)）
