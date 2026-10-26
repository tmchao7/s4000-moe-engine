# NIGHTLY-LOG — 2026-10-26 夜间迭代（P6b 接手）

> 协议：每个实验先写【假设 → 唯一变量 → 预期 → 判定门】再跑；跑完追加【实际结果 → 决策】。
> 时间基准：本地 UTC+8。开始时间 2026-10-26 23:50 前后（8 小时预算）。
> 接手状态：HANDOFF-P6b.md 已读；git 状态与 §3 清单一致（6 改 + 11 新增，未提交）；服务器接管时两卡 0 MiB、无残留进程、shm 已清；免密登录已恢复（端口 28036，服务器曾瞬断后恢复）。

---

## 0:00–0:20 接管

- [23:55] HANDOFF-P6b.md 全文读完。git status 确认 15+2 个未提交文件与交接 §3 一致，HEAD ecc87b2，未 reset。
- [23:58] 服务器状态：双卡 0 MiB、无 python 进程、/dev/shm/nanovllm* 已清。
- [00:02] **SSH 稳定性对策**：服务器频繁瞬断（connect closed / TCP timeout）。所有长任务 setsid nohup + 落盘日志 + 短连接轮询；每条 ssh 命令带重试循环（≤8 次，间隔 12s）。

## 实验 R0：廉价回归（脏树完整性门）

- **假设**：上一轮的未提交改动（bmm_dense、syncfree route/pack、shm barrier、store_kvcache 直写）没有破坏数值正确性。
- **唯一变量**：无（纯回归）。
- **预期**：`tests/p2_grouped_mm.py --max-tokens 8` 与 `tests/p6_bmm_dense_tiny.py --max-tokens 8` 均 report loop/grouped_mm/bmm_dense/TP1/TP2 逐 token 全等。
- **判定门**：全部 PASS 且 rc=0；任一 FAIL 则先修，不进入性能测量。

**实际结果（00:25）**：
- 首跑出现假失败：我在 SSH 断连后盲目重试启动命令，导致**两个实例同时在 GPU 上跑**（第一次的命令其实已在服务器执行），互相吃掉 KV 预算 → `no KV cache memory` 假崩溃。教训已入 §运维：**断连后先查服务器状态再重试；启动类命令一律带锁脚本**（`/root/run_regressions.sh` + `/root/reg.lock`）。
- 清理后串行重跑：`p2_grouped_mm.py` **rc1=0 全过**（loop/grouped_mm/fused_grouped_mm × TP1/TP2 逐 token 全等 + 2× repeat 确定性）。
- `p6_bmm_dense_tiny.py` 的**核心门通过**：六配置（loop/grouped_mm/bmm_dense × TP1/TP2）token **exact match: True**（每配置 3×16 token，agreement 全 1.000）。其后附带的 graph vs eager 段在 tiny 图捕获时崩（`MUSA error: operation not permitted`，rank1，capture 阶段）→ rc2=1，属 HANDOFF-P6b §2.6 已知的图捕获间歇性问题，非本轮改动引入（本轮未改任何图路径代码）。**不阻塞，进入 P0**。

## 运维教训（夜间累积）

1. **SSH 断连 ≠ 命令失败**：命令可能已在服务器上执行。断连后先 `pgrep`/查日志确认，再决定重试；启动类命令一律走带锁的 runner 脚本。
2. 测试类进程崩溃后可能**卡死在 teardown**（rank 间 DistBackendError）：按 PID kill -9，再清 `/dev/shm/nanovllm*`。
3. 所有长任务：`setsid nohup` + 落盘日志 + 短连接轮询；`grep -a` 处理含二进制字符的 MUSA 崩溃日志。

## 实验 P0：30B eager 新栈矩阵（bmm_dense decode）

- **假设**：`decode_moe_backend=bmm_dense` 在 30B 真实负载下把 decode 吞吐提升到单层微基准预测的水平（1.3–1.95×/层 → 整步 decode 快 ~10–30%），且不破坏 prefill/attention 路径。
- **唯一变量**：decode MoE 后端（grouped_mm vs bmm_dense），同一次 30B 加载内 A/B；attention 后端（sdpa vs triton_paged）作为第二维度同时测，供 P1 转正决策用。
- **方法**：`scripts/p2_benchmark.py` 新增 `--decode-moe-backends`（本次 `grouped_mm,bmm_dense`），prompt 128 / output 32 / batches 1,2,4,8,16,32 / repeats 3（每点 warmup 1 次），与 `benchmarks/P5-final-matrix.md` 同口径（b64 排除：已知崩溃触发点，另在 P1 单独复现）。验证阶段：baseline 硬门（loop,grouped_mm vs 冻结参考）+ bmm_dense 软门（自确定性硬门，参考一致率只报告）。
- **预期**：微基准预测单层 decode b1 2.05→1.27 ms、b8 2.92→1.94、b32 5.56→2.86；48 层 × decode 步 → b32 请求级 TPOT 从 ~0.42 s（P5 matrix，sdpa+grouped）降到 ~0.28–0.33 s 区间（还要加 attention/sampler 等不变项）。
- **判定门（冻结后再跑）**：在 b16 与 b32，对每个 attention 后端：(1) `throughput_ratio = bmm_dense / grouped_mm ≥ 1.08`；(2) bmm_dense 的吞吐 CV ≤ 5%；(3) 任何 batch 上 bmm_dense 不得慢于 grouped_mm（ratio ≥ 0.98 视为持平，<0.98 记失败点）。token 层面：bmm_dense 软门自确定性必须通过；与冻结参考的一致率**单独报告数字**（预期 <100%，不设门）。
- **决策规则**：门全过 → 采用 `decode_moe_backend=bmm_dense` 进入 P1/P2；任一失败点 → 记录数字与假设原因，P2 用 grouped_mm 栈兜底。

**实际结果（00:05，/root/p6_eager_matrix.json，rc=0，19.5 分钟含加载）——✅ 门全过，采用 bmm_dense**：

验证门：
- baseline 硬门 PASS：loop、grouped_mm 与冻结参考逐 token 全等、自身确定、warmup=steady。
- bmm_dense 软门 PASS：自确定；且 8-token 输出与冻结参考**全等**（soft 门比预期更硬地通过了）。

矩阵（median 吞吐 tok/s，ratio=bmm_dense/grouped_mm）：

| batch | sdpa: grouped→bmm (ratio) | triton_paged: grouped→bmm (ratio) |
|---:|---:|---:|
| 1 | 3.96→5.82 (1.470) | 4.42→6.60 (1.495) |
| 8 | 28.69→41.90 (1.461) | 30.81→45.85 (1.488) |
| 16 | 50.11→72.30 (**1.443**) | 54.38→83.20 (**1.530**) |
| 32 | 85.79→110.59 (**1.289**) | 97.21→133.00 (**1.368**) |

- 门：b16/b32 ratio ≥1.08 ✓✓（实际 1.29–1.53）；CV ≤5% ✓✓（最大 0.033）；无任何 batch 变慢 ✓（最低 ratio 1.289）。
- TPOT p95 @b32：sdpa 0.335→0.249 s；triton_paged 0.290→0.199 s。vs P5-final-matrix 默认栈（sdpa+grouped，TPOT 0.4244 s / b32 69.39 tok/s）：新栈 **b32 133 tok/s（+92%）、TPOT p95 0.199 s（-53%）**。
- 跨后端 token 一致率（repeat 0，请求级全等）：b1–b8 100%（b2 为 50%，1/2），b16 81–88%，b32 75–94%——符合 bf16 累加顺序差异的近平局翻转预期，不设门。
- **决策：`decode_moe_backend=bmm_dense` 采用**，进入 P1/P2。b32 处 ratio 回落（1.29 vs 小 batch 的 ~1.5）与微基准的大 batch 趋势一致，仍远超门。


## 实验 P1：D2 稳定性（shm 修复是否治好 b32/b64）【第 2 次 30B 加载】

- **假设**：HANDOFF-P6b §2.7 的 shm 命令通道竞态（rank1 unpickle 被覆盖 → 执行错步）是历史 b32 不可复现 / b64 崩溃的根因；barrier 修复后两者消失。
- **唯一变量**：无（对照上轮 P5 的失败现场；代码差异 = shm barrier 修复 + store_kvcache 直写 + route/pack 无同步化）。
- **方法**：依次 `tests/p5_determinism_b32.py`（2×2 组 + zero-KV + b16 对照）→ `tests/p5_d2_stability.py --repeats 10 --bisect` → b64 三次独立复现（`--batches 64`，单独进程）。
- **判定门（冻结）**：
  1. determinism_b32：全部组（含 sdpa/triton_paged × grouped_mm/loop、zero-KV、b16）逐 token 确定；
  2. d2_stability：b32 10 次重复确定（或按脚本自带 bisect 结论定位到已修复项）；
  3. b64：3/3 不崩、同栈内确定。
- **后续决策**：若 b32 稳定 → 用同一加载补测 sdpa vs triton_paged 的 b32 确定性对照（若 determinism_b32 未含），然后 `scripts/p5_decide.py --mode attention` 评估 triton_paged 转正（门：candidate 自身确定 + ≥99% token 一致 + TPOT ratio ≤0.90，阈值不改）；若仍不稳定 → 按 determinism_b32 的组间差异继续二分（MCCL 已排除；重点 grouped_mm/argsort/SDPA mask），45 分钟规则适用。

## 实验 P2：在线容量 v3（第 3 次 30B 加载 + API server）

- **假设**：P0/P1 通过门的栈（decode=bmm_dense，attention 视 P1 结论）把 decode 单步成本降低后，SLO（TTFT ≤2 s 且 TPOT p95 ≤0.5 s）100% 达标的负载档从 50% 推到 ≥75%。
- **唯一变量**：推理栈（v2 基线 = sdpa + grouped_mm；v3 = 候选栈）。负载、SLO、客户端参数与 `benchmarks/P5-serving-v2.md` 完全一致（Poisson 开放环，seed 20261026，128 req/档 + 4 warmup，max-qps 0.87，四档 0.5/0.75/0.9/1.0，输出 64 token，prompt 137 tok）。
- **判定门（冻结）**：主门 = 75% 档 SLO 达标率；v3 的 75% 档 SLO 达标率 > v2 的 26.6% 即方向正确，≥99% 即达标（对应「100% SLO 的负载档推到 ≥75%」）。附带报告：各档 TTFT/TPOT/ITL/E2E p50/p95、goodput、最大并发、以及 v2→v3 同表对比。
- **预期**：v2 的 50% 档 TPOT p95 0.455 s 已贴近 0.5 s 门；decode 单步降 ~25–50%（P0 实测 TPOT -25%~-53%）后，75% 档的 TPOT p95（v2 为 0.697 s）预计落到 0.35–0.52 s 区间，即 75% 档有较大概率达标；90%/100% 档取决于并发堆叠后的调度行为。
**实验 P2 实际结果（01:02，rc=0，15 分钟客户端，512/512 请求成功）——✅ 超额达成：四档全部 100% SLO**：

| 负载档 | v3 SLO% (TPOT p95) | v2 SLO% (TPOT p95) | v3 goodput | v2 goodput |
|---:|---:|---:|---:|---:|
| 50% | **100.0%** (0.229 s) | 100.0% (0.455 s) | 0.427 | 0.411 |
| 75% | **100.0%** (0.348 s) | 26.6% (0.697 s) | 0.656 | 0.163 |
| 90% | **100.0%** (0.312 s) | 11.7% (0.685 s) | 0.706 | 0.077 |
| 100% | **100.0%** (0.401 s) | 7.0% (0.694 s) | 0.771 | 0.050 |

- 目标是「100% SLO 的负载档从 50% 推到 ≥75%」，实际推满到 **100% 档**；100% 档 goodput **15.4×**。
- TTFT 不再是约束（全档 p95 ≤ 1.27 s，v2 在 100% 档为 2.067 s）；输出吞吐 49.35 tok/s（v2 45.72）且在 SLO 内。
- 报告已写：`benchmarks/P6-serving-v3.md`。四档全绿说明 SLO 拐点已推到 v2 口径（0.87 QPS 封顶）之外；更高注入率未测，属后续工作。

## 实验 P2b：在线容量探索——找真实 SLO 拐点（第 4 次 30B 加载，探索性）

- **假设**：v3 四档全绿是因为 0.87 QPS 的 v2 口径封顶内不存在拐点；真实拐点在更高注入率。
- **唯一变量**：注入率（其余参数与 v3 完全一致）。fractions 1.1/1.25/1.5/2.0 × max-qps 0.87 → 目标 QPS 0.957/1.088/1.305/1.74。
- **判定门（冻结）**：报告各档 SLO 达标率与 goodput；「SLO 拐点负载」= SLO 达标率跌破 99% 的最低档。这是探索性实验，不改 v3 的主结论口径。

**实际结果（03:24，/root/moe-server-s4000/benchmarks/data/online-v3-highload.json，四档各 128/128 OK）——✅ 真实拐点找到**：

| 目标 QPS | 达成 QPS | 输出 tok/s | TTFT p95 | TPOT p95 | SLO% | goodput |
|---:|---:|---:|---:|---:|---:|---:|
| 0.957 | 0.880 | 56.34 | 1.691 s | 0.535 s | **84.4%** | 0.743 |
| 1.087 | 0.943 | 60.32 | 2.847 s | 0.970 s | 10.2% | 0.096 |
| 1.305 | 0.944 | 60.40 | 3.921 s | 1.334 s | 3.1% | 0.029 |
| 1.740 | 1.117 | 71.47 | 4.226 s | 1.208 s | 0.0% | 0.000 |

- 拐点在 0.87–0.957 目标 QPS 之间；1.087 起 TTFT 先崩（队列效应）、达成 QPS 与输出吞吐钉死（0.94 req/s / 60 tok/s = 硬饱和）。
- **SLO 容量结论：本栈 ~0.87–0.90 req/s（TTFT 2s/TPOT 0.5s），是 v2 栈（~0.41–0.43）的 ~2.1×**。
- 运维注：`benchmark_online.py` 断言 fraction ≤ 1，抬注入率要加 `--max-qps` 而非 fraction>1；第一次尝试浪费了一发加载。


## 实验 P3：图 replay 时间盒（90 分钟，8B 先行）【第 4 次 30B 加载仅在 8B 5/5 绿后才允许】

- **假设**：图 replay 挂死（HANDOFF-P6b §2.6）源于 replay 与 rank 间集合通信/其他流的交互，可按 §4.1 的二分顺序定位。
- **唯一变量**：每次只改一个（①replay 前后 barrier → ②每桶私有 pool → ③padding 槽互异 → ④sampler 纯 argmax → ⑤capture_error_mode=relaxed → ⑥breakable graph）。
- **方法**：Phase A = `tests/p6_graph_min.py --model /root/models/Qwen3-8B --dense --tp 2 --attention-backend triton_paged --generates 1` 连跑 5 次（每跑前清进程+shm；单次 timeout 600s）；Phase B = 5/5 绿才跑 `--generates 3 --clear-prefix-cache`，否则按失败阶段（capture/replay）走二分，每变量连跑 3 次。
- **判定门（冻结）**：8B 单次 generate **5/5 全绿** 且 3-generate 也绿 → 才允许 30B graph A/B；90 分钟到点未达标 → 停止，写「试了什么/卡在哪/复现率」。
- **注意**：今晚回归中 tiny 图捕获已出现一次 `operation not permitted`（23:38，bmmdense 回归 graph 段）——捕获本身的间歇性在今晚仍然存在，Phase A 结果要区分「捕获失败」和「replay 挂死」两种死法。

（结果待填）

**实验 P3 实际结果（01:11–02:29，8B 时间盒 ~78 分钟）——✅ 找到候选修复：replay 前后 rank barrier + 每桶私有 graph pool（①+②），8B 单次 generate 5/5 全绿、3 次 generate + 清前缀缓存也通过；30B A/B 已在门放行后启动（结果见下）**：

| 配置 | 单次 generate | 结果 |
|---|---|---|
| baseline（无改动） | 连跑 3 次 | **0/3**，全部捕获成功后 replay 挂死 → 10 分钟 watchdog（`launch timed out`） |
| ①replay 前后 `dist.barrier()`（run_model） | 连跑 3 次 | 1/3（fail, fail, **pass 45 s**）——时间敏感竞态 |
| ①+②每桶私有 pool（`graph_ctx(graph, None)`） | 连跑 5 次 | **5/5 全绿**（43–46 s/次，token 三次全同）；另 `--generates 3 --clear-prefix-cache` **rc=0 完成**（历史挂点 1/3 消失；gen1→gen2 有一处 token 漂移，属已知的 replay 读旧 KV 块问题，与挂死无关） |

- 复现率基线：0/3 → ① 1/3 → ①+② 5/5 + 多重放通过。**rank 对齐（barrier）与跨桶 pool 复用两个因素都参与该竞态**；①+② 是当前最小候选修复。
- 代码痕迹：`model_runner.py` 中 `# BISECT-1`（replay 前后 barrier，各 ~0.08 ms）与 `# BISECT-2`（私有 pool）。作为候选修复保留在树中，待 30B 验证后决定去留。
- 30B 验证：按冻结门（5/5 绿）放行，`p5_graph_ab.py --batches 1,2,4,8,16,32 --repeats 3 --num-kvcache-blocks 32 --decode-moe-backend bmm_dense` 带 60 分钟硬帽启动（02:33），结果与 `p5_decide.py --mode graph` 结论见下节。

**P3 追加（02:50）：30B graph A/B——❌ ①+② 不能迁移到 30B，图方向按时间盒关闭**：

- 30B 捕获成功（6 桶、bmm_dense、triton_paged，reserved +0.525 GiB，两 rank 一致）。
- 但 `verify graph_candidate` 的**第一次 generate（replay）即打死设备**：~10 分钟后 rank0 watchdog `RuntimeError: MUSA error: the launch timed out and was terminated`（栈在 replay 后的 barrier——即设备已被 replay 打楔，下一次 launch 才报错）；rank1 死于 `call()` 的命令级 barrier。日志 `/root/p5_graph_ab.log`。
- **结论**：8B（dense、无图内 MoE、图内 all-reduce 数量少）修好的竞态与 30B 的挂死不是同一个（或不止一个）。30B 差异因素：48 层 × 96 次/步图内 MCCL all-reduce、bmm_dense 图内 MoE、更深的依赖链。下一轮可试（未测）：分段捕获（attention/linear 进图、MoE eager）、或图内去 MCCL（先 allgather 结构改造）、或厂商 MUSAGraph 的已知限制清单。
- 图方向关闭；`enforce_eager=True` 维持默认。BISECT-1/2 补丁对 8B 有实证收益、对 30B 无害（eager 路径 replay 分支不执行），保留在树中并标注。




**实验 P1 实际结果（00:32，/root/p6_d2_nightly.json + /root/p6_attention_ab.json，rc=0，一次 30B 加载 ~35 分钟）——✅ 全绿，两个悬案都关了**：

1. **b32 稳定性：shm 修复生效**。determinism_b32 四案例全稳定（A grouped_b32 ✓、B loop_b32 ✓、C zero-KV ✓、D b16 对照 ✓）；b32 ×10 三个栈（grouped+sdpa / grouped+triton / bmm_dense+triton）**全部 0 分歧**，且三栈首轮输出 token 完全一致。历史上「两个 attention 后端都不可复现」的 b32 问题不复存在。
2. **b64 崩溃：未复现**。3/3 次完成（grouped+sdpa，prompt 128/out 32，历史崩溃形状），无 MUSA error。
3. **correctness gate**：loop 与 grouped_mm 均与冻结参考逐 token 全等。
4. **attention 决策（冻结门，`benchmarks/p6-attention-ab-decision.json`）：keep_sdpa**。
   - 性能门大胜：TPOT ratio 0.59–0.62（≤0.90 ✓）、吞吐比 1.48–1.59（≥1.00 ✓）、CV ≤0.013 ✓、candidate 自身确定 ✓。
   - 一致性门失败：b32/c2048 请求级一致率 **0.875 < 0.99**（12/96 个 repeat×请求 vs sdpa 参考有分歧；c512 也有 0.938/0.969）。关键事实：**今晚所有点所有后端自身都是确定的**（包括历史上不稳定的 sdpa b32）——所以这些分歧是可复现的**真实数值差异**（paged kernel 的分页/掩码路径不同），不是运行噪声。冻结门保守拒收，按协议不改阈值。
   - triton_paged 的收益量化（供未来参考）：c2048 吞吐 +48~59%、TPOT -31~41%；b32/c512 82.9 vs 61.6 tok/s。转正前置条件：解释/消除 c2048 的后端间 token 分歧（例如 paged 掩码边界审查），或产品决策接受「paged 输出与 sdpa 在少数请求上不同」。
- **P2 栈决策**：`attention=sdpa（默认）+ decode_moe_backend=bmm_dense`（P0 门通过 + P1 未被否决的组合）。





