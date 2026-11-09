# P30 下一阶段机会评估（2026-11-09）

> 本文件回答：在 P29 已把 TP/EP/PP/SP/DP/量化/RR 都探过之后，**结合当前项目特性（2×S4000、30B MoE、产品化服务）还剩什么值得做**。
> 结论先行：**先补部署硬化，再做 8k 长上下文产品化；不建议再追 kernel/DP2/量化。**

## 1. 当前项目特性（做决策的约束）

- 硬件：2×MTT S4000，48GB/卡；30B bf16 约 56.9GiB，单卡放不下，必须 TP2；
- 模型：Qwen3-30B-A3B-FP8，128 专家 top-8，48 层，4 个 KV 头；生产 TP2+EP2；
- 生产栈：`triton_paged` + `grouped_mm` prefill + `bmm_dense` decode + eager + `stall_free` + chunk 256；
- 产品：`s4000-server`、OpenAI API、systemd/install/smoke/release、16/16 回归；
- P29 已判：30B DP2/FP8 保留/8-bit/4-bit/RR 放置均 NO-GO 或不可用；PP/SP/CP 在 2 卡上 ROI 不成立。

## 2. 候选机会矩阵（我的估算）

| 候选 | 用户影响 | 工作量 | 风险 | 建议 |
|---|---|---|---|---|
| **A. 8k 长上下文产品化** | 高（真实长文档/知识库） | 2–3 天 | 中：8k TTFT ~14s，质量门未做 | **P0，第二优先** |
| **B. 部署/发布硬化** | 高（能不能真正上线） | 1 天 | 低 | **P0，第一优先** |
| C. 容量/可观测性 | 中（SLO/容量规划） | 1–2 天 | 低 | P1 |
| D. OpenAI API 兼容（tools/embeddings） | 中，取决于产品范围 | 2–3 天 | 低 | P2，明确非目标也可 |
| E. PD 分离产品化 | 中高（ITL） | 高 | 高：2 卡与 TP2/EP2 冲突 | 暂不做 |
| F. 自定义 prefill attention / 融合 | 不确定 | 高 | 高：图被 MCCL 阻塞，历史融合收益低 | 不做 |
| G. 多副本 + LB | 对 30B 低（放不下单卡） | 1–2 天 | 低 | 只作为 0.6B 运维能力，已完成基础脚本 |
| H. 量化/DP2/RR | 已被实测否掉 | — | — | 不重开 |

## 3. 推荐顺序：B → A → C

### B. 部署/发布硬化（1 天，先做）

为什么先做：功能再多，**不能稳定部署就没有用户价值**。当前还有几个“文档写了但没端到端验证”的点：

1. **`--version` 程序名**：已修。原来 `python -m` 打印 `api_server.py 0.1.0`，现在统一 `s4000-server 0.1.0`；新增 CLI 测试（服务器 7 passed）。
2. **systemd 实际启动 30B**：目前只做了 `systemd-analyze verify`，没有真正 `systemctl start` 30B；需要一次冷启动 + `/health` + 一条 chat + restart/stop。
3. **installer venv 分支**：服务器没有 `python3-venv`，之前只验证了 `--target` fallback；需要一台有 venv 的机器验证 venv 分支。
4. **release 包安装 + 0.6B smoke**：构建/内容断言已有，但“从 tarball 安装后起服务”应再跑一次。

验收：30B systemd 起停、0.6B tarball 起服务、健康/chat/stop、双卡 0 MiB。

**Day1 进展**：已修复 target 安装的 transformers/tokenizers 版本冲突；release tarball + `--target` 安装 + `serve_s4000.sh` + 30B health/chat/stream 已实跑通过；systemd 实启与 venv 分支受当前实例限制，未验证（见 [day1 报告](../benchmarks/P30-day1-deploy-hardening.md)）。

**Day2 进展**：8k 合成 needle-QA **9/9 PASS**（7615 tokens，3 code × 3 depth，整批 42.2s）；API 超长输入返回 400；8k 并发/SLO 探针已写但被交接中止，**没有 SLO 数字**（见 [day2 报告](../benchmarks/P30-day2-longctx-quality.md)）。下一任先重跑 SLO。

**Day3 进展**：SLO 探针重跑完成——18×7600 并发 TTFT p50 63.5 s（超 2 s 交互门 32×），聚合 prefill ~2.1k tok/s（单流 ~3.9×）。**判决：8k = 离线/批处理能力，非交互能力**（与本节预估一致）；用法已写入 OPERATIONS §6，P30-A 验收四项全齐（见 [day3 报告](../benchmarks/P30-day3-longctx-slo.md)）。A 项到此收口，后续只剩 C（容量/可观测性）已有基础上的按需补测。

### A. 8k 长上下文产品化（2–3 天，第二优先）

为什么值得：Day8 已证明 30B 在 8k 不 OOM，decode 210.8ms；这是**真实差异化能力**，而且不需要新 kernel。

当前缺口：

1. **质量门没做**：Day8 用随机 token，只证明 kernel/显存不炸；需要真实长文档 QA/needle 评测（建议 ≥32 条，阈值先定再跑）。
2. **交互 SLO**：8k prefill ~14s；如果产品是聊天，这不可接受；如果是离线/批处理，可以接受。需要先决定场景。
3. **API 输入校验**：`--max-model-len 8192` 已可配；要确认 `prompt+max_tokens` 超限返回 400，而不是排队/截断。
4. **并发容量**：8k 时 651 blocks → 约 20 条并发；要压测 SLO，而不是只看单条。

验收：真实 8k 质量门、20 并发压测、错误输入 400、文档写清“opt-in”。

### C. 容量/可观测性（1–2 天，第三优先）

- 项目已有 `/metrics` + Prometheus 指标；需要审计是否覆盖 **KV 使用率、queue depth、TTFT/TPOT、prefill/decode 比例**；
- 做容量规划：4k vs 8k 的 max_num_seqs/KV 上限；
- 这一项适合部署后按真实流量补，不必先做。

## 4. 为什么不推荐其他方向

- **E PD 分离**：P15 单机 P1D 有数据，但当前 30B 已经 TP2 用满两卡，PD 需要重新切分/双实例，复杂度和收益不匹配；
- **F 自定义 prefill/fusion**：历史 fused RMSNorm −0.8%、bmm 容量 −10% 门 FAIL、整图被 MCCL 阻塞；没有新证据前不重开；
- **G 多副本**：0.6B 启动/健康/停止脚本已完成，但 30B 单卡放不下，不能当 30B 扩展方案；
- **H 量化/DP2/RR**：P29 已用实测否掉；不要为了填时间重做。

## 5. 如果只做一件事

做 **B 部署硬化**。原因：A 的 8k prefill 14s 是否可接受还没定，而 B 是所有后续工作的前提；B 工作量小、风险低、能直接提升“可交付”。

## 6. 面试口径（“你接下来会做什么”）

> 我不会继续堆 kernel。P29 已经把 TP2+EP2 之外的路都量过了：DP2 缺 FP8 保留、W8/4-bit 没有速度、RR 过不了 token 门、PP/SP 在 2 卡上不划算。下一步先把部署链路真正跑通（systemd/installer/release），再把 8k 长上下文做成有质量门和 SLO 的 opt-in 能力；量化/DP2 等厂商 MATE 或 FP8 支持再说。

## 7. 证据索引

- [P29-FINAL.md](P29-FINAL.md)：P29 总结/面试
- [day8 长上下文](../benchmarks/P29-day8-longctx.md)
- [day9 多副本](../benchmarks/P29-day9-dp2-ops.md)
- [OPERATIONS.md](OPERATIONS.md)：运维/多副本