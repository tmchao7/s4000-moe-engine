# CLOSEOUT — 收官清单:已解决 / 无法解决(P14,2026-10-30)

> 本文只做一件事:把"还剩什么"一次说清。技术细节以 [ISSUES.md](./ISSUES.md) 为准,当前状态以 [HANDOFF.md](./HANDOFF.md) 为准。

## A. 本轮已解决(P14)

| 项 | 做法 | 证据 |
|---|---|---|
| 客户端跨 repeat 暖缓存口径伪象(I8) | `benchmark_online.py --prompt-seed-per-repeat`(默认关) | 单测 `test_prompt_seed_per_repeat_changes_unique_prompts`;ISSUES I8 |
| I12 teardown 卡死(引擎侧) | `LLMEngine.exit()` 看门狗:`NANOVLLM_EXIT_TIMEOUT_S`(默认 60 s)超时 kill rank、unlink shm、`os._exit` | `tests/p14_exit_watchdog.py` 三例 PASS;ISSUES I12 |
| "~2048 图上限"是不是我们的误判 | 否:SGLang PR #27537 + torchada 文档均确认 MUSA 驱动每进程约 2048 个 `musaGraphExec_t` | 工单 §2 已补 |
| I29 有没有可用线索 | 有:MCCL 官方文档点名"捕获/未捕获 MCCL 混用 → 死锁",`MCCL_GRAPH_MIXING_SUPPORT=1` 使 0.6B TP2 整图完整通过 0/11→3/9 | [benchmarks/P14-mccl-mixing.md](../benchmarks/P14-mccl-mixing.md) |
| eager 容量上界 | 拐点落在 1.6×~1.8×：1.8× 三 repeat 94.5/87.5/80.5%（<99% 门），2.0× 30–62%；硬饱和吞吐 ≈1.2 req/s；已验证 ≥1.070 req/s | [P14-serving.md](../benchmarks/P14-serving.md) |
| 1k 冷态 1.0 档 | chunk-2048 + seed-per-repeat：TTFT p50 1.08–1.12 s，**门 PASS**（chunk-1024 为 2.2 s）；代价：slo_strict 18–24%（最大 ITL ~1.2–1.5 s） | 同上 |

## B. 无法解决(及原因 / 谁能解决 / 当前绕过)

| 条目 | 为什么我们修不了 | 谁能解决 / 需要什么 | 当前绕过 |
|---|---|---|---|
| **I1/I2/I29/E4 引擎级整图 replay 间歇楔死** | 闭源 driver/runtime/MCCL 层。P14 证实"MCCL 混用"是成分之一(缓解 0→33%),但剩余楔死源无诊断工具可见 | 摩尔线程:确认单通信器混用是否同属已知死锁;提供图级 dump/调试 | `enforce_eager=True`(生产默认);分段图仅单流可显式开 |
| **图可执行上限 ~2048 / torch_musa 直连无轮换** | 驱动限制;torchada 才有 LRU 轮换 | 厂商给官方口径;或我们改用 torchada(超出本项目范围) | 分段图 dense 桶 1056 图/rank 在上限内 |
| **E3 `torch._grouped_mm` 无旋钮、不可捕获** | 厂商 op | 厂商 | decode 用 bmm_dense;prefill 保持 grouped_mm |
| **E5 厂商 mp22 fast path 数值错误** | 厂商 kernel(rel 1.3–341、b64 NaN) | 厂商 | `use_mp22=False` |
| **E6 launch watchdog 10 min、报错点漂移** | 驱动参数,不可配 | 厂商 | 自研硬 watchdog(`--verify-watchdog-s`)+ P14 的 exit 看门狗 |
| **I28 torch.musa 跨块 strided view D2H 返回垃圾** | 平台 bug 嫌疑 | torch_musa 团队 | `.contiguous()`/分 K/V 读 |
| **I31 30B 首 batch 与稳态逐位不同** | 机制未查(候选:JIT/autotune/allocator 首次布局)。无外部线索,也非性能问题 | 需单独开一轮 bisect;收益低 | 一致性门丢弃首 batch |
| **I19 每步固定成本 / S6 EAGLE3** | I19 已被实测否定"launch-bound→融合换步时";EAGLE3 需训练头 + 大改引擎,且前置收益不明 | 立项决策(周级工程) | 不做 |
| **PD 分离** | 需第二台机器与跨机网络（两机 iperf3/RDMA 带宽未测）。**单机 M0 探针已做**：本机有 ConnectX-6（mlx5_0/1/bond）、`mt_peermem` 内核模块已加载（GPUDirect RDMA 前置满足）；`mthreads-gmi topo -m` 显示 GPU0/1 ↔ NIC0/1 为 NODE（同 NUMA），NIC2 为 SYS；`mooncake-transfer-engine-musa 0.3.13.post1` wheel 可从源下载（未安装，不改 site-packages）| 用户提供第二台机器 | 调研文档 `docs/PD-DISAGG-SURVEY.md` |
| **I20 int8 KV** | 周级工程。探针结论：朴素 torch dequant 比 bf16 拷贝慢 5.7×，必须融进 attention kernel | 立项 | 探针数据见 ISSUES I20 |
| **E1/E2 kernel/MoE 配置自调** | 可做但 ROI 低(decode 已用 bmm_dense) | 有人愿意花 30–60 min | 不做 |

## C. 需要用户亲自做

1. **发厂商工单**:[E-vendor-ticket.md](./E-vendor-ticket.md)(已补 MCCL 文档线索、2048 交叉印证、P14 探针结果)。外发必须由你来。
2. 是否 push 本地 commit(本仓库一直未 push)。
3. 是否授权 30B 整图 A/B(红线,需另行批准;P14 结果显示即使开 `MCCL_GRAPH_MIXING_SUPPORT=1` 仍有楔死风险,不建议)。
