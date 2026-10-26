# HANDOFF-P6c — 夜间迭代定案：bmm_dense 落地、b32/b64 治好、SLO 容量推满、图 8B 修复/30B 未修

> 写于 2026-10-27 凌晨（夜间自主迭代收尾）。**本文档是 HANDOFF-P6b 的后继**：P6b 的所有【待验证】项都已验证，本文只记录新增事实与下一步。逐实验过程与预冻结门在 `docs/NIGHTLY-LOG.md`。
> 当前栈状态：`attention=sdpa`（默认）、`moe_backend=grouped_mm`（prefill）、**`decode_moe_backend=bmm_dense`（新事实上的最优 decode 栈）**、`enforce_eager=True`（默认不变）、scheduler 默认。

## 0. 30 秒

1. **`decode_moe_backend=bmm_dense` 已通过 30B 全部门并落地**：离线 b16/b32 吞吐比 1.29–1.53×，CV ≤3.3%，验证门（含 8-token 冻结参考全等）全过。在线四档全 100% SLO。
2. **历史 b32 不可复现 / b64 崩溃 = shm 命令通道竞态，已证实并修复**（P6b §2.7 假设成立）：30B 上 b32 ×10 三个栈 0 分歧、b64 3/3 不崩、连历史上不稳定的 sdpa b32 都稳了。
3. **在线容量（Poisson 开放环，SLO=TTFT 2s/TPOT 0.5s）**：v2 只有 50% 档 100% SLO；v3（唯一变量=decode 后端）**50/75/90/100% 四档全 100% SLO**，100% 档 goodput 0.050→0.771 req/s（15.4×）。
4. **triton_paged 未转正（keep_sdpa，冻结门）**：性能大胜（TPOT ratio 0.59、吞吐 +48~59%），但 b32/c2048 请求级一致率 0.875 < 0.99。今晚所有后端自身都确定 → 分歧是 paged kernel 的可复现数值差异（页边界/掩码），非噪声。
5. **图 replay：8B 找到候选修复，30B 未修**。8B（dense）：baseline 0/3 挂死 → `replay 前后 rank barrier + 每桶私有 graph pool` **5/5 全绿 + 3 次 generate/清前缀缓存通过**。30B：捕获成功（6 桶 +0.52 GiB）但第一次 replay 仍打死设备。两处不同源。图方向按时间盒关闭，`enforce_eager=True` 维持。

## 1. 代码变更（相对 P6b 的工作区，全部未提交，见本地 git status）

- `scripts/p2_benchmark.py`：新增 `--decode-moe-backends`（逗号列表，同加载内 A/B decode 后端；records/summary 带 `decode_moe_backend` 字段）；`run_verification` 支持 `decode_backend` + soft 模式（自确定性硬门、参考一致率只报告）。
- `nanovllm/entrypoints/openai/api_server.py`：新增 `--decode-moe-backend` 透传（P2 v3 用它）。
- `nanovllm/engine/model_runner.py`：`# BISECT-1`（run_model replay 前后 `dist.barrier()`）与 `# BISECT-2`（capture_cudagraph 每桶 `graph_ctx(graph, None)` 私有池）——8B 候选修复，eager 默认路径不受影响，30B 图未修但无损。
- `tests/p6_d2_nightly.py`（新）：一次 30B 加载跑完 D2 全套（det cases + b32×10 三栈 + attention A/B + b64），A/B 输出与 p5_attention_ab 同 schema，可直接喂 `p5_decide.py --mode attention`。
- `benchmarks/P6-serving-v3.md`、`benchmarks/p6-attention-ab-decision.json`、`benchmarks/P6-nightly.md`（新报告）。
- 默认值**未改**：门没过的（attention、graph）一律维持原默认。

## 2. 新证据清单（/root/ 与本地 benchmarks/data/p6b/ 各一份）

| 文件 | 内容 |
|---|---|
| `p6_eager_matrix.json/.jsonl` | P0：30B 矩阵（2 attention × 2 decode × 6 batch），验证门 + 软门全过 |
| `p6_d2_nightly.json/.jsonl` | P1：det cases A–D、b32×10 三栈、attention A/B、b64×3 |
| `p6_attention_ab.json` + `benchmarks/p6-attention-ab-decision.json` | A/B 数据 + keep_sdpa 决策 |
| `online-v3.json` / `online-v3-highload.json` | P2/P2b 在线容量（v3 四档全绿；highload 定位真实拐点 0.87–0.957 QPS、硬饱和 0.94 req/s / 60 tok/s） |
| `p6_graph_5x.log`、`p6_graph_bisect1.log`、`p6_graph_bisect2.log`、`p6_graph_3gen_bisect12.log` | P3：0/3 → 1/3 → 5/5 + 多重放 |
| `p5_graph_ab.log` | 30B：捕获成功 + 第一次 replay 打死设备的完整现场 |

## 3. 下一轮建议（优先级序）

1. **triton_paged c2048 分歧定位**（离线已量化 +48~59% 吞吐）：审 paged kernel 页边界/掩码；或给出 bf16 logits diff 上界说明"输出差异可接受"，由产品决策转正。门不要动。
2. **30B 图换路子**：分段捕获（MoE eager）/ 图内去 MCCL / 厂商 MUSAGraph 限制清单。8B 修复不可迁移，不要重走 barrier/私有池。
3. **P2b 已完成**：真实拐点 0.87–0.957 目标 QPS（0.957 档 SLO 84.4%），硬饱和 0.94 req/s / 60 tok/s；SLO 容量 ≈ v2 的 2.1×。后续如抬容量：接入 triton_paged（待一致性门）或降 prefill 排队（TTFT 先崩）。
4. 前缀缓存 cold/hot 探针（P5 遗留，`P5-final-matrix.md` §3 有现成命令）。
5. `p5_graph_ab.py` 的 verify 阶段可加 per-step watchdog（30B 失败时 rank1 先死、rank0 卡满 10 分钟才 watchdog，浪费且难定位）。

## 4. 运维教训（本轮新增）

- **SSH 断连 ≠ 命令失败**：断连后先 `pgrep -fa` 查服务器再重试；启动类命令一律带锁脚本（本轮示范：`/root/run_p0.sh` 等）。未锚定 `pgrep -f` 自杀又踩了一次（`run_p3a` 模式匹配到执行命令的 bash 自己）——模式必须 `^/bin/bash ...` 或 `^python3 ...` 或 `[s]pawn_main`。
- 崩溃进程可能卡死 teardown（rank 间 DistBackendError）：按 PID kill -9 + `rm -f /dev/shm/nanovllm*`。
- MUSA 崩溃日志含二进制字符：`grep -a` / `strings`。
- 服务器重启后 SSH 端口会变（本轮 18484→28036）：以用户给的最新地址为准，更新 `~/.ssh/config`。
- 30B 模型页缓存热后加载只要 ~5–8 分钟（非冷启动 13–15 分钟）。
