# 厂商工单包:MUSA Graph 引擎级 replay 楔死 + 图可执行上限求证(I29/E4/E6/I32-②)

> 状态:待发送(2026-10-29 夜整理;2026-10-30 P14 补充官方文档线索、2048 交叉印证、MCCL_GRAPH_MIXING_SUPPORT 探针数据;**外发须用户亲自操作**)。发送渠道:摩尔线程开发者论坛 developer.mthreads.com / vllm-musa GitHub issues。
> 我们是自研推理引擎(基于 nano-vllm fork),非商业客户;但报告带最小复现与分级探针。官方文档称 S4000 上 MUSA Graph 默认开启,与我们观测的引擎级楔死矛盾——这正是要请厂商澄清的核心。

## 0. 环境版本

| 项 | 值 |
|---|---|
| 硬件 | 2× MTT S4000(48 GB,PCIe Gen5 x16),TP2 |
| 软件 | torch 2.11.0.post1 + torch_musa 2.11、MCCL 2.11、Triton(mtgpu)、Ubuntu 22.04(容器) |
| 驱动/运行时 | 实例 uptime 5d+,未被重启;驱动版本随工单附 `mthreads-gmi` 全文 |

## 1. 问题一:引擎级 MUSAGraph+MCCL replay 间歇性楔死(I29)

**现象**:双卡 TP2 引擎在 graph replay 路径上间歇性 rendezvous 楔死。崩溃形态三种(对称双 rank 自旋 / 单 rank 死+另一 rank 旋 / watchdog 10 min 后杀且报错点漂移)。非确定复现:同一配置有时全绿。

**分级探针结果**(`tests/p9_mccl_graph_probe.py`,每 case 独立进程+watchdog):

| case | 结果 |
|---|---|
| 纯张量图:96 次图内 all-reduce、b32 payload、48×bmm+AR、~2400 节点、六桶共享 pool、完整复刻引擎 capture/replay 协议 | **全绿,零楔死** |
| TP1 引擎级(0.6B/8B) | 从未楔死 |
| **0.6B TP2 + paged attention,引擎级首次 replay** | **3/3 楔死(最小复现,加载仅 2 min)** |
| 0.6B TP2 + sdpa | 撞 2 次 generate 后楔死 |
| replay 前后显式 barrier | 无效 |

**界定**:纯张量图(含大量图内 MCCL)是好的;引擎级才坏 → 必要成分 = **引擎 replay 协议**(每步 fill 输入缓冲/scratch slot 填充/replay 后贴 eager compute_logits 与采样节奏)× **TP2 集合通信**。请厂商指认:这个组合里哪一步踩了 MCCL/runtime 的未定义行为?

**官方文档线索(P14 补充)**:[MCCL 使用 MUSA Graphs](https://docs.mthreads.com/musa-sdk/version-5.2.0/libraries/mccl/mccl_user_guide_2.4/usage/musagraph.zh-CN/) 写明"被捕获与未捕获的 MCCL 操作混用,且多个通信器从同一线程 `musaGraphLaunch()` 时,已观察到 MUSA 死锁",并给出开关 `MCCL_GRAPH_MIXING_SUPPORT`(默认 0,见 [环境变量表](https://docs.mthreads.com/musa-sdk/version-5.2.0/libraries/mccl/mccl_user_guide_2.4/env/))。我们的引擎正是"图内 AR + replay 后紧跟 eager AR(embed_head / compute_logits)"的混用形态,但只有一个 WORLD 通信器,与文档描述的"多通信器"并不完全吻合。**P14 探针结果见 §1.1**。请厂商澄清:该死锁机制是否也覆盖单通信器的 captured/non-captured 混用?

**附带请求**:CUDA 生态有 nsys / cudaGraphDebugDotPrint;MUSA 侧我们找不到等价物,这也是定位慢的根因。torchada 的 `TORCHADA_CUDA_GRAPH_DEBUG_DUMP_PATH`(.dot 转储)已是一个方向——**torch_musa 直连路径(不经 torchada)是否有同样的图转储/图节点级诊断?**

### 1.1 P14 探针结果(MCCL_GRAPH_MIXING_SUPPORT)

0.6B TP2 paged 整图最小复现,独立进程 + 硬 timeout,交错多轮:

| 臂 | 完整通过(5/5 generate,与 eager 逐位一致,退出正常) |
|---|---|
| 不设变量 | 0 / 8(楔死,多卡在第 1–4 次 generate) |
| `MCCL_GRAPH_MIXING_SUPPORT=0` | 0 / 3 |
| `MCCL_GRAPH_MIXING_SUPPORT=1` | **3 / 9**(单 run 36 s) |

结论:打开混用支持能缓解但不能消除楔死,**说明"图内/图外 MCCL 混用"是成分之一、尚有另一个楔死源**。请厂商:① 确认单通信器(WORLD)的 captured/non-captured 混用是否也在该死锁机制覆盖范围内;② 指认剩余楔死源或提供图级诊断。详细数据:`benchmarks/P14-mccl-mixing.md`。

## 2. 问题二:MUSA 驱动每进程存活图可执行文件 ~2048 上限(torchada 已披露,求证细节)

torchada README:"The MUSA driver caps the number of live graph *executables* per process (~2048)";torchada 用模板 + LRU 轮换(默认 1900,replay 被逐出图时从模板重新实例化,~0.3 ms/次)解决。

**P14 交叉印证**:[SGLang PR #27537](https://github.com/sgl-project/sglang/pull/27537)(2026-06 合入)同样写明"MUSA 驱动每进程约 2048 个 `musaGraphExec_t`,PCG 路径实例化过多导致 capture 失败",靠 torchada ≥0.1.59 的 LRU 轮换规避——厂商生态已知,不是我们的误判。

求证:① 上限的准确数值与版本演进?② **torch_musa 直连路径是否有等价的轮换机制**?(我们直连 torch_musa 2.11,未装 torchada)③ 我们实测:35 桶 × 3360 图在捕获 1632/3360 处 MUSA OOM(图池 5.91 GiB 时)——OOM 报错之外,是否还叠加了图可执行上限?④ 1056 图/rank(11 桶 × 48 层 × 2 段)的规模是否有已知风险?

## 3. 问题三:`torch._grouped_mm` 不可捕获是否预期

捕获期报 `operation failed due to a previous error during capture`(连带 `operation not permitted when stream is capturing`)。我们已改用 dense bmm 家族做 decode MoE(bmm_dense,可捕获、数值正确)。请确认:grouped_mm 路线是否有捕获计划?厂商 vLLM-MUSA 的 MoE 图内路径用的是什么算子家族?

## 4. 问题四:MUSA launch watchdog 诊断能力(E6)

设备楔死后 watchdog 10 分钟才杀,且报错点漂移(报在下一个 barrier,不在真实现场)。已用自研硬 watchdog(`--verify-watchdog-s`,超时 `os._exit(97)`)自救。请求:更短的可配置 watchdog、报错点对齐、以及 §1 的图级诊断。

## 5. 我方证据清单(随工单附)

- `tests/p9_mccl_graph_probe.py`(分级探针,独立进程+watchdog)
- `tests/p8_graph_padfix.py --tp 2`(引擎级复现 harness,两进程口径)
- `benchmarks/P9-graph.md`、`benchmarks/P11-graph.md`、`benchmarks/P12-serving.md`(全过程与冻结门)
- 分段图替代实现作为 workaround 已验证:`NANOVLLM_SEGMENT_GRAPH=1`(图内零 MCCL,每层两段,all-reduce eager 在段间)——0.6B/8B/30B 离线门全过、768 图零失败、跨臂逐位 0 翻转;但在 TP2 上只能作为 workaround,批量 ≥8 时因桶 padding 与段间 eager AR 净负(详见 P12)。
- 运行环境:`docs/musa-pitfalls.md` 记录的其他平台坑(跨块 strided view D2H 返回垃圾 I28、FP8 逐 expert 反量化 351 s 等)。

## 6. 姿态与诉求

我们不需要商业支持,只求:① 引擎级 replay 楔死的已知限制/修复计划;② 图可执行上限的官方口径与 torch_musa 直连轮换路线图;③ 图级 debug/dump 工具。若 ①短期无解,分段图(图内零 MCCL)是我们的既定路线,请帮忙审阅其 capture/replay 姿势是否符合 runtime 预期(代码 ~200 行,`model_runner.py::capture_segment_graphs`)。
