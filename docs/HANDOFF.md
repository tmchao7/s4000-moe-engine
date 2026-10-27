# HANDOFF — moe-server-s4000（MUSA S4000 双卡推理引擎）

> 最后更新：2026-10-26（会话 #4：P5 指标/内核 + attention A/B + 上游 MoE 复用评估 + P6 graph 可行性）
> **本文件是给下一位接手的 AI（DeepSeek/GPT）的唯一事实来源**。规则：**文档与代码/测试冲突时以代码和测试为准**，修完文档再继续；不要凭记忆补命令、补数字、补机制。
>
> **2026-10-26 20:25 追加（会话 #5 收手交接）**：P6 图捕获**已被证明在算子层面可行**（旧结论错误，见 `docs/P6-graph-capture.md`），当前唯一硬阻塞是厂商 `vllm_fused` MoE 在 decode 时一次性申请 ~15.6 GiB 导致捕获前 warmup OOM。**接手请先读 `docs/HANDOFF-P6-next.md`**（含 T1–T4 任务、精确命令、验收门、10 条踩坑记录），再回到本文件看全局。
>
> **2026-10-26 22:55 追加（会话 #6 收手交接，最新）**：**先读 `docs/HANDOFF-P6b.md`，它推翻上面 20:25 的 T1 结论**。15.6 GiB 是 `qwen3_moe.py` 的惰性权重转置副本（576 MiB/层），不是厂商 workspace；零拷贝厂商 MoE 已验证但默认配置慢 2.3–6.3×，**decode 改用新的 `bmm_dense`（1.3–1.95× 于 grouped_mm，且可捕获）**；30B 图捕获成功（6 桶，+0.34 GiB）但 **replay 在模型规模上间歇性挂死**（唯一未解）；另修掉 shm 命令通道竞态（很可能就是历史 b32/b64 不稳定根因）。夜间迭代主线是 eager 的 `bmm_dense + triton_paged`，图只做时间盒诊断。
>
> **2026-10-27 追加（自检+快修后定稿，最新）**：**先读 `docs/HANDOFF-P6c.md`（交接定稿：当前栈、成绩单、18 项问题逐项状态表、SOP、红线）**。核心事实：`decode_moe_backend=bmm_dense` 过全部 30B 门并落地（离线 b32 110.6/133 tok/s，+59%/+92%）；b32/b64 不稳定被 shm 修复治好（证实）；在线容量 v3 四档全 100% SLO（100% 档 goodput 15.4×）、真实天花板 0.94 req/s / 60 tok/s（容量 2.1× v2）；triton_paged 冻结门判 keep_sdpa（0.875<0.99，可复现数值差异）；图 replay 8B 已修（barrier+私有池，5/5 绿）但 30B 未修。问题台账：`docs/P6-open-issues.md`（18 项，4 项已修），逐实验过程 `docs/NIGHTLY-LOG.md`。代码 HEAD `91e3fd3`，未 push。
>
> **2026-10-27 追加（I4 logits 定位，最新）**：I4 的 TP2 同状态 logits 诊断已完成（代码 `41ab19b`）：扩展页边界/poison-tail 参考全绿，首层差仅 `2.44e-4`，但 bf16 深层放大到最大 logits 差 6.09，并在 8,928 个 request-step 中出现 19 个近平局 top-1 翻转。故**不是页表/mask bug，不做无依据内核试错，不改冻结门，默认继续 SDPA**；详情先读 `docs/P6-I4-attention-logits.md`，再读 `HANDOFF-P6c.md`。
>
> **2026-10-27 追加（P6 覆盖/图隔离收口，最新）**：**先读 `docs/HANDOFF-P6c.md`**。生产保持 `sdpa + grouped_mm prefill + bmm_dense decode + enforce_eager`；I4 继续 keep_sdpa。I5 b64 稳定性与 I6 prefix-cache 命中已验证关闭；图隔离两控制组均 0/5，未跑 30B 图 A/B；I7 是高方差软边界而非精确 knee；I8 仍在单一远端 runner 中。`fcfb81c` 已推送，禁止“不要 push”的旧规则已废止；以 `git ls-remote` 核验收尾提交。

---

## 0. 30 秒开工

```bash
ssh -p 28036 root@connect.sha1.seetacloud.com        # 免密；中继偶发瞬断，长任务必须 setsid nohup
cd /root/moe-server-s4000                            # 代码（服务器无 .git）
# 先确认环境干净，再跑便宜的门（见 §6），最后才碰 30B（每次加载 ~10 分钟）
```

本地仓库 `feat/musa-port` 是唯一源头（origin = github.com/tmchao7/moe-server-s4000）。**不 push**，除非用户明确要求。

---

## 1. 事实清单（不要猜，照抄）

| 项 | 值 |
|---|---|
| 服务器 | `root@connect.sha1.seetacloud.com -p 28036`（AutoDL 容器） |
| 代码目录 | `/root/moe-server-s4000`（**没有 .git**） |
| 30B 模型 | `/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8`（加载期反量化为 bf16，约 29.5 GiB/卡） |
| 8B 模型 | `/root/models/Qwen3-8B`（dense 基线，P1 用） |
| tiny 模型 | `/tmp/tiny_moe`（float32，2 层 8 专家）与 `/tmp/tiny_moe_bf16`（由 `tests/p5_vllm_fused_tiny.py` 生成） |
| 冻结参考 | `benchmarks/p2-reference-8tok.json`（4 条 prompt × 8 token，确定性构建） |
| 节点规模 | 单机 2× MTT S4000 48GB，PCIe Gen5 x16，无 NVLink |
| 软件栈 | torch 2.11 + torch_musa 2.11、MCCL、triton 3.2.0（backend `mtgpu`）、transformers 5.13、Python 3.10 |
| 默认后端（不得擅改） | `attention_backend=sdpa`、`moe_backend=grouped_mm`、`expert_placement=linear`、`enforce_eager=True` |

### 当前 commit 与证据

- 代码：最近三个提交 `293b02e`（metrics/serving v2）、`7e0c451`（kernels/profiling/A-B/上游 MoE 复用）、以及本文件所在的 docs 提交。
- profiling 原始证据：`benchmarks/data/p5_profile.json`（服务器 `/root/p5_profile.json`），SHA256 `147e7761799863017210d74f5da236cc07f027971c4eb7afc4f95fe7bdbc089d`。
- 决策记录：`benchmarks/p5-profile-decision.json`（profiling 路线）、`benchmarks/p5-attention-ab-decision.json`（attention A/B，见 §3）。

---

## 2. 项目现在做到哪

| 阶段 | 状态 | 一句话 |
|---|---|---|
| P0 基线 | 完成 | vllm-musa + Qwen3-8B TP2：8.4 tok/s（c=32，不扩展） |
| P1 MUSA 移植 | 完成 | Qwen3-8B 单卡 113.5 tok/s @b64，首 token 与 HF 对齐 |
| P2 MoE 双卡 | 完成 | 30B-A3B TP2+EP2；grouped-MM 74.0 tok/s @b32、120.4 @b64；比 loop 快 3.0–3.2×；贪心可复现 |
| P3 调度 | 完成 | FCFS vs stall-free 在线对比（mixed step 0→124），短 prompt 收益个位数 |
| P4 服务化 | 完成 | OpenAI 兼容 API + 开放环压测（128 请求 100% 成功） |
| P5 指标/内核 | 进行中 | schema v2 + Prometheus + 增量 detokenizer + 设备 profiling + Triton paged decode + 上游 MoE 复用评估；**attention A/B 见 §3** |
| P6 graph capture | **已判死（有证据）** | MUSA 上 `torch._grouped_mm` 与 Triton kernel 不可捕获（MCCL 可捕获），详见 `docs/P6-graph-capture.md` |

### 已完成的实测结论（不要重跑）

- **profiling（第 1 发）**：长上下文 b32 时 attention 占 forward 58.56%、MoE 计算 53.48%、通信 5.45%、rank 不平衡 5.19% → paged attention 第一优先；**通信重叠与专家放置被否决**（低于 20%/10% 门）。
- **上游 MoE 复用**（`docs/P5-moe-optimization-survey.md`）：vLLM 的 Triton `fused_experts` 可直接调用，**token 与 grouped_mm 24/24 一致**；但默认配置下更慢（1024 token 262 ms/层），且 vllm-musa 的厂商 fast path（mp22）在 1024 token 直接 illegal memory access → 已集成但**保持 opt-in**，不改默认。
- **手写融合**（`fused_grouped_mm`）在 bf16 tiny 上 22/24 token 一致（近并列翻转）→ 数值忠实度不如上游 kernel，不作为默认。
- **graph capture**：plain op ✅、MCCL all_reduce ✅、`torch._grouped_mm` ❌（"operation not permitted when stream is capturing"）、Triton paged kernel ❌ → P6 取消。

---

## 3. 加载台账（每一发都按此执行）

| # | 内容 | 状态 | 产物 |
|---:|---|---|---|
| 1 | profiling 矩阵（8 个 batch×context 点） | ✅ 完成 | `/root/p5_profile.json`、`benchmarks/P5-profiling.md`、`benchmarks/p5-profile-decision.json` |
| 2 | SDPA vs Triton paged A/B | 🔄 运行中/见下 | `/root/p5_attention_ab.json`、`benchmarks/P5-attention-ab.md`、`benchmarks/p5-attention-ab-decision.json` |
| 3 | MoE 三路 A/B（grouped / fused_grouped / vllm_fused） | ⏸ 未做（可选） | `/root/p5_moe_ab.json` + `--mode moe` 决策 |
| 4 | 终局 schema-v2 矩阵 + 前缀缓存冷热 | ⏸ 未做 | `/root/p5_final_matrix.json`、`benchmarks/P5-final-matrix.md` |
| 5 | 在线 Poisson 容量/goodput | ⏸ 未做 | `benchmarks/P5-serving-v2.md` |
| ~~6~~ | ~~graph capture A/B~~ | ❌ 取消 | 证据在 `docs/P6-graph-capture.md` |
| 7 | P0：30B eager 新栈矩阵（2 attention × 2 decode × b1..b32，同加载 A/B） | ✅ 完成 | `/root/p6_eager_matrix.json/.jsonl`、`benchmarks/P6-nightly.md` §1；**bmm_dense 采用**（b16/b32 比 1.44–1.53/1.29–1.37） |
| 8 | P1：D2 合并验证（det cases A–D + b32×10 三栈 + attention A/B + b64×3） | ✅ 完成 | `/root/p6_d2_nightly.json/.jsonl`、`benchmarks/p6-attention-ab-decision.json`（**keep_sdpa**）；b32/b64 均已治好 |
| 9 | P2：在线容量 v3（sdpa + bmm_dense，50/75/90/100% 四档） | ✅ 完成 | `benchmarks/data/online-v3.json`、`benchmarks/P6-serving-v3.md`；**四档全 100% SLO**（100% 档 goodput 15.4×） |
| 10 | P2b：高负载拐点探索（`--max-qps 1.74`，fractions 0.55/0.625/0.75/1.0） | ✅ 完成 | `benchmarks/data/online-v3-highload.json`、`benchmarks/P6-nightly.md` §4；真实拐点 0.87–0.957 目标 QPS，硬饱和 0.94 req/s / 60 tok/s |
| 11 | 30B graph A/B（8B ①+② 5/5 绿后按门放行） | ❌ 第一次 replay 仍打死设备（捕获成功） | `/root/p5_graph_ab.log`、`benchmarks/P6-nightly.md` §5；图方向时间盒关闭 |

夜间台账总账见 `docs/NIGHTLY-LOG.md`（每发预冻结门 + 实际结果）；交接 successor 文档 `docs/HANDOFF-P6c.md`。

### 第 2 发（attention A/B）——如果还在跑或要重跑

```bash
ssh -f -n -p 28036 root@connect.sha1.seetacloud.com \
 'cd /root/moe-server-s4000 && setsid nohup env PYTHONPATH=/root/moe-server-s4000 \
  python3 scripts/p5_attention_ab.py --workloads 16x512,32x512,16x2048,32x2048 \
    --output-len 32 --repeats 3 --attention-backends sdpa,triton_paged \
    --output /root/p5_attention_ab.json > /root/p5_attention_ab.log 2>&1 < /dev/null &'
```
- 先看日志：`[p5-profile] 30B correctness gate PASS`，然后逐点 `[p5-ab] ... tok/s=... mismatches=...`。
- 结束后**必须**跑决策（阈值已冻结在 `scripts/p5_decide.py`）：
```bash
PYTHONPATH=/root/moe-server-s4000 python3 scripts/p5_decide.py --mode attention \
  --input /root/p5_attention_ab.json \
  --output benchmarks/p5-attention-ab-decision.json \
  --markdown benchmarks/P5-attention-ab.md
```
- `adopt_triton_paged` → 才允许改 `nanovllm/config.py` 与 `api_server.py` 的默认 `attention_backend`；`keep_sdpa` → 记录负结果，默认不动。

### 第 3 发（MoE 三路 A/B，可选）

```bash
python3 scripts/p2_benchmark.py --batches 8,16,32 --prompt-len 128 --output-len 32 \
  --repeats 3 --moe-backends grouped_mm,fused_grouped_mm,vllm_fused \
  --attention-backends <第 2 发的决定> --output /root/p5_moe_ab.json
PYTHONPATH=/root/moe-server-s4000 python3 scripts/p5_decide.py --mode moe \
  --input /root/p5_moe_ab.json --output benchmarks/p5-moe-ab-decision.json \
  --markdown benchmarks/P5-moe-ab.md
```
⚠️ `vllm_fused` 走 generic Triton 路径时 prefill 很慢（1024 token ≈262 ms/层），跑之前先确认时间预算；若只想要数据不必三路全跑，可用 `grouped_mm,fused_grouped_mm`。

### 第 4 发（终局矩阵 + 前缀缓存）

```bash
python3 scripts/p2_benchmark.py --batches 1,2,4,8,16,32,64 --prompt-len 128 --output-len 32 \
  --repeats 3 --moe-backends <决定> --attention-backends sdpa,<决定> \
  --probe-batches 8 --probe-prompt-len 512 --probe-shared-prefix-len 384 \
  --probe-cache-modes cold,hot --output /root/p5_final_matrix.json
python3 scripts/render_benchmark.py /root/p5_final_matrix.json \
  --output-dir benchmarks/data --accept-batches 16,32 --markdown-out benchmarks/P5-final-matrix.md
```
注意：前缀缓存按 **256-token 整块**命中，所以 `--probe-shared-prefix-len` 必须 ≥256（脚本会断言）。

### 第 5 发（在线容量）

```bash
# 服务（用最终决定的默认栈）
python3 -m nanovllm.entrypoints.openai.api_server --model <30B> \
  --served-model-name qwen3-30b-a3b --scheduler-policy stall_free \
  --attention-backend <决定> --moe-backend <决定>
# 就绪后（curl /health）：
PYTHONPATH=/root/moe-server-s4000 python3 scripts/benchmark_online.py --model qwen3-30b-a3b \
  --tokenizer /root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --max-qps <由第 4 发 b32 吞吐 ÷64 ×0.8> --requests 128 --max-tokens 64 --prompt-words 128 \
  --arrival-pattern poisson --slo-ttft-ms 2000 --slo-tpot-ms 500 \
  --fractions 0.25,0.5,0.75,0.9 --output benchmarks/data/online-v2.json
```

---

## 4. 冻结门（写死在代码里，禁止"看结果再改阈值"）

| 模式 | 门 | 代码 |
|---|---|---|
| `attention` | token 全等；context=2048 且 batch∈{16,32} 的 p95 TPOT 比值 ≤0.90；吞吐比 ≥1.00；CV ≤5% | `scripts/p5_decide.py::ATTENTION_GATES` |
| `moe` | token 全等；b16、b32 吞吐比均 ≥1.08；p95 TPOT 比 ≤1.00；CV ≤5% | `MOE_GATES` |
| `graph` | token 全等；b1 ≥+15%、b8 ≥+5%；b32/b64 ≥0.98；CV ≤5%；显存增量 ≤2 GiB | `GRAPH_GATES` |
| profiling | attention ≥30%；MoE ≥35% 或 route/combine ≥10%（MoE 内）；通信 ≥20%；不平衡 >10% | `scripts/p5_profile.py::decide` |

决策 JSON 一律包含 `thresholds / observed / decision / evidence_sha256`。**边界用例已单测**：`tests/p5_decide.py`。

---

## 5. 禁止事项（上一位 AI 最容易被坑的地方）

1. **不要改默认后端**，除非对应模式的冻结门通过并在决策 JSON 里写清。
2. **不要重跑第 1 发 profiling**（结果已冻结、原始 JSON 有 sha256）。
3. **不要把 legacy v1（P2/P3/P4）指标与 v2 混比**：v1 的 `tpot` 字段实为 token-gap ITL、CV 用 `pstdev/median`。见 `docs/P5-metrics-audit.md`。
4. **不要声称 graph capture 可用**：已实测不可捕获；也不要改 `enforce_eager` 默认。
5. **不要声称移植了 MoonEP/UniEP/MegaMoE**：只借融合划分；理由与实测见 `docs/P5-moe-optimization-survey.md`。
6. **不要用厂商 mp22 fast path**：1024 token 会 illegal memory access；`vllm_fused` 里已强制 `use_mp22_fused_moe=False`。
7. **不要在 30B 上跳过正确性门**直接测性能：`p2_benchmark` 默认开启，`--skip-verification` 只在明确知道原因时使用。
8. **不要在一个 30B 进程还在跑时启动第二个**（显存 48GB/卡，两个必挂）。
9. **不要用 `from scripts.x import`**：这台机器的 site-packages 里有一个同名 `scripts` 包会把它覆盖，必须按文件路径导入（见 §6）。
10. **不要相信 `scp` 与长 SSH 会话**：用 tar-over-ssh 与 `setsid nohup`。

---

## 6. 环境与运维

### 同步代码（本地 → 服务器）

```bash
cd <local repo> && COPYFILE_DISABLE=1 tar -czf - \
  --exclude=.git --exclude=.zcode --exclude=__pycache__ --exclude='benchmarks/data' . \
| ssh -p 28036 root@connect.sha1.seetacloud.com \
  'cd /root/moe-server-s4000 && tar -xzf - && find . -name "._*" -delete && python3 -m compileall -q nanovllm tests scripts && echo COMPILE_OK'
```

### 启动长任务（SSH 断了不影响）

```bash
ssh -f -n -p 28036 root@connect.sha1.seetacloud.com \
 'cd /root/moe-server-s4000 && setsid nohup env PYTHONPATH=/root/moe-server-s4000 \
  python3 <script> <args> > /root/<name>.log 2>&1 < /dev/null &'
# 轮询用短连接；不要 sleep 在 ssh 里
ssh -p 28036 ... 'tail -5 /root/<name>.log; ps -o pid,etime -p <pid>'
```

### 清理仪式（每次测试前后）

```bash
pkill -9 -f "[s]pawn_main"; pkill -9 -f "tests/p[0-9]_"; pkill -9 -f "api_serve[r]"
rm -f /dev/shm/nanovllm*
mthreads-gmi | grep -E "^[01] .*MiB"     # 必须 0MiB
```
`pkill -f` 会自匹配（远端 shell 命令行含模式）→ 用括号技巧 `"fo[o]"`。

### 本机特有的坑（血泪）

| 坑 | 症状 | 处理 |
|---|---|---|
| site-packages 有同名 `scripts` 包 | `from scripts.p5_profile import ...` → `ModuleNotFoundError` | 按文件路径导入（`sys.path.insert(0, str(Path(__file__).parent))` + `from p5_profile import ...`） |
| 30B 加载 ~10 分钟 | 日志长时间为空、显存已到 29.5 GiB | 显存到位≠加载完，等 `[p5-profile] ... gate PASS` 或 `[bench verify]` |
| vllm-musa mp22 fast path | `illegal memory access`（≥1024 token）或 `K must be divisible by AlignmentA/B` | 强制 `use_mp22_fused_moe=False`，或不要用 `vllm_fused` |
| tiny 模型形状 | vllm fused MoE 的 muDNN 要求 tile 对齐，tiny 会 abort | tiny 验证用 generic Triton 路径；生产形状才用 fast path |
| 前缀缓存 | 短 prompt 完全无命中 | `shared_prefix_len ≥ 256`（整块）才可能命中 |
| Triton `.jit` 不能写在 `python -c` 里 | `OSError: could not get source code` | 必须写成 `.py` 文件运行 |
| 中继拉大文件会截断 | 50KB JSON 偶发 0 字节 | 传/拉都带重试并校验大小 |
| `useset -e` 下的管道 | 退出码丢失 | 用 `set -o pipefail` |

---

## 7. 代码地图

```
nanovllm/
├── config.py                   # 运行配置：TP/EP、moe_backend、attention_backend、调度、metrics
├── async_engine.py             # 后台线程引擎 + 命令队列 + Prometheus + 有界队列(429)
├── benchmarking.py             # TPOT/ITL/CV/goodput/percentile 的唯一定义处
├── prometheus_metrics.py       # counter/gauge/histogram 导出
├── incremental_detokenizer.py  # 增量 detokenize（UTF-8 边界、stop 回退）
├── profiling.py                # MUSA Event 分段计时 + observe 计数
├── kernels/
│   ├── paged_attention.py      # Triton paged GQA decode（head_dim 128 / page 256，runtime loop）
│   └── moe_fusion.py           # 手写 SwiGLU + 确定性 combine（对照用）
├── engine/
│   ├── llm_engine.py           # step 循环、指标、JSONL trace
│   ├── scheduler.py            # FCFS / stall_free / stall_free_lpm + ScheduledBatch
│   ├── block_manager.py        # paged KV + block-hash 前缀缓存 + 计数
│   ├── model_runner.py         # musa 设备、shm 多进程、prefill/decode/mixed、graph 骨架（禁用）
│   └── sequence.py             # 序列状态 + 请求级指标
├── entrypoints/openai/api_server.py  # /health /v1/models /v1/chat/completions /metrics
├── models/qwen3_moe.py         # packed experts、loop/grouped/fused/vllm_fused 四种后端、确定性 top-k
└── layers/attention.py         # prefill SDPA + decode 双后端（sdpa / triton_paged）
scripts/  p5_profile.py, p5_attention_ab.py, p5_decide.py, p2_benchmark.py,
          benchmark_online.py, render_benchmark.py, profile_experts.py
tests/    p2_*（正确性）, p3_*（调度/指标）, p4_*（服务）, p5_*（指标/内核/决策/上游复用）, p6_graph_probe.py
docs/     P5-metrics-audit.md, engine-vs-vllm-sglang.md, P5-moe-optimization-survey.md,
          P6-graph-capture.md, architecture.md, roadmap.md, musa-pitfalls.md
benchmarks/ P2-moe-tp2.md, P3-scheduler.md, P4-serving.md, P5-profiling.md,
          p2-reference-8tok.json, p5-profile-decision.json, data/（原始 JSON，不入库）
```

---

## 8. 先跑哪些测试（顺序不要乱）

```bash
# CPU，秒级
PYTHONPATH=/root/moe-server-s4000 python3 tests/p5_metrics_v2.py
PYTHONPATH=/root/moe-server-s4000 python3 tests/p5_detokenizer.py
PYTHONPATH=/root/moe-server-s4000 python3 tests/p4_api.py
PYTHONPATH=/root/moe-server-s4000 python3 tests/p4_benchmark_client.py
PYTHONPATH=/root/moe-server-s4000 python3 tests/p5_decide.py
# 设备，无需模型
PYTHONPATH=/root/moe-server-s4000 python3 tests/p5_paged_attention.py
# tiny（每次 1–2 分钟，注意清理）
PYTHONPATH=/root/moe-server-s4000 python3 tests/p2_grouped_mm.py
PYTHONPATH=/root/moe-server-s4000 python3 tests/p5_vllm_fused_tiny.py
PYTHONPATH=/root/moe-server-s4000 python3 tests/p3_scheduler.py
PYTHONPATH=/root/moe-server-s4000 python3 tests/p4_async_engine.py
PYTHONPATH=/root/moe-server-s4000 python3 tests/p6_graph_probe.py      # 预期 FAIL（已知阻塞）
```

---

## 9. 下一步候选与进入条件

| 候选 | 进入条件 |
|---|---|
| 改默认 attention 后端 | 第 2 发决策 `adopt_triton_paged` |
| 改默认 MoE 后端 | 第 3 发决策 `adopt_*`（token 全等 + ≥8% + TPOT 不退化 + CV） |
| 上游 MoE 的 shape-driven autotune（`get_moe_configs` 风格） | 需要一次独立加载做配置 sweep；只有 sweep 后 ≥8% 才值得再谈默认 |
| route/pack 融合 | 先用低开销计时复核 44.9% 是否被 event 插桩放大 |
| graph capture 重启 | 需要一个可捕获的 MoE（或 MUSA runtime 修复）；MCCL 已确认可捕获 |
| radix tree / 投机解码 / 多实例 | 与当前主线无关，别做 |

---

## 10. 交接自检清单（动手前逐条打勾）

- [ ] 能 `ssh` 上服务器，`cd /root/moe-server-s4000` 成功
- [ ] `git log --oneline -3` 能看到本轮提交；本地与远端 md5 一致
- [ ] 两卡 0MiB、无残留 `spawn_main`/`api_server`
- [ ] §8 的 CPU 门全绿
- [ ] 读过 §5 禁止事项
- [ ] 知道本次要花几发加载、每发结束**先落盘提交再做下一发**
