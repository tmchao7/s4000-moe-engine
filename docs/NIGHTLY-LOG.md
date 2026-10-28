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






---

# P8 — 2026-10-27 23:00 起 24h 轮次日志

> 规则:每个实验先写门再跑;一次只跑一个 GPU 任务;失败/负结果照样记录。H0=2026-10-27 23:00 (+0800),H24=2026-10-28 23:00。

### H0–H1.5 P0 基线核验 + KV 块数一致化 (I21) + 指标 v3

- **起点核验**:本地 HEAD `c0d1a7023970`(=交接文档起点),工作区干净(仅 .DS_Store)。服务器双卡 0 MiB、无 python/spawn、/dev/shm 干净、无 lock。本地↔远端五个关键文件 md5 一致,无需同步基线。
- **I21 修复**(`model_runner.py` allocate_kv_cache):启发式算出 blocks 后,TP>1 时 `dist.all_reduce(MIN)` 统一两 rank 的 `config.num_kvcache_blocks`,再打印 `[kvcache]` 行与分配。
- **冻结门(跑前写)**:① 两 rank `[kvcache] blocks=` 相同;② `tests/p6_bmm_dense_tiny.py` 六配置逐 token 全等、rc=0。
- **实际**:`[bmm-tiny] exact match across all six configs: True`,`rc=0`;TP2 两配置下 rank0/rank1 blocks=343080/343080、341994/341994(本加载内两 rank 启发式恰好相同,divergence 钳制路径未触发,但 int64 all_reduce(MIN) 在 MCCL 上验证可用;真 divergence 待 30B 首次加载确认)。
- **判定**:PASS。
- **证据**:`/root/p8_tiny_regression.log`(远端)。
- **决策**:进入 P0.3。

- **指标 v3 实现**:`nanovllm/benchmarking.py`(wilson_ci / token_goodput / classify_error / conforming / percentile method="linear")、`scripts/benchmark_online.py`(schema v3:error_kind 分桶、--request-timeout-s[wait_for 取消→服务端断连 cancel]、slo_standard+slo_strict 双 SLO、--repeats + Wilson CI、--mode concurrency、--prompt-mode mixed 3:4:3、GPU 每 2s 采样、p99_reference_only、queue_p50/p95 来自服务端 final-chunk metrics)、`api_server.py`(stream_options.include_metrics 时 final chunk 带 queue/ttft/e2e/tpot/itls)、`tests/test_benchmarking_v3.py`。
- **门**:`python3 -m pytest tests/test_benchmarking_v3.py` 全绿;旧 v2/v4 JSON 回放不抛异常。
- **实际**:15 passed(含 v4-paged.json 回放);kv_prefix_hit_tokens_total 服务端已存在(block_manager.metrics)无需新增。
- **判定**:PASS。

---

### H1.75– P1.1 fused RMSNorm — 冻结门(跑前写,不可跑后改)

- **假设**:0.6B b1 每步 2,401 launches 中 ~1,000 来自 RMSNorm 的 fp32 上采样逐元素链;单 Triton kernel(bf16 进出、fp32 内累加、同时产 residual)可把 decode 步时降 ≥8%。
- **唯一变量**:`NANOVLLM_FUSED_RMSNORM` 开关(运行时切换,同进程内交替);其余栈不动(eager、sdpa、b1、fcfs、max_num_seqs=1)。
- **预期**:步时中位数 −8~15%(63ms 基线 → 54–58ms);launch/步 −800~−1,100。
- **冻结门**:
  - G1(L1,kernel 级):合成 bf16 输入上 fused vs eager 输出 **fp32 cast 后 max_abs <2e-3**(residual / no-residual 两路径);bf16 输出逐元素不等数仅报告(预期 ≤1 ulp 边界翻转)。依据:L1 口径是单层 fp32 参考误差(I4 先例),端到端 bf16 logits 的 1-ulp 量化噪声不可用绝对阈判定。
  - G2(L2,端到端):0.6B b1、8 prompts × 32 步 free-decode,top-1 翻转率 <0.5%(按可比步数),**所有**翻转处 (top1−top2) margin <0.5;token 流一致数报告。首个翻转后的步不比 logits(轨迹已分叉)。
  - G3(自确定):两模式各 3 次重复逐 token 一致。
  - G4(性能):交错 baseline/fused 各 5 trial × 32 decode 步,**中位数的 median** fused ≤ 0.92×baseline,**两组 CV ≤5%**,否则 INVALID。
  - G5(launch 证据):profiler 单步 launch_count(fused) ≤ launch_count(eager) − 800。
- **判定规则**:G1–G3 全过 + G4 过 → PASS(flag 默认开,进 tiny 回归);G4 不过 → 记数字、flag 默认关;CV>5% → 增至 10 交错或判 INVALID。
- **证据**:`/root/p8_fusion.json`、`/root/p8_fusion_ab.log`。

#### P1.1 fused RMSNorm A/B — 实际结果(H2.2)

- **实际**:
  - G1 PASS:三组形状(hidden 1024/2048、head_dim 128)fp32_max_abs=0.0、**bf16 输出 0 元素不等**(bitwise 全等)。
  - G3 PASS:两模式各 3 次重复逐 token 全自确定。
  - G5 PASS:launch 2,402 → 1,442(−960/步);kernel device 时间 31.87 → 21.14 ms(**−10.7 ms/步,−33%**)。
  - G2 FAIL:prompt 6 step 4 翻转,margin=1.375(≥0.5 门);可比 228 步 1 翻转(0.44%)。与 G1 bitwise 全等**矛盾**(0.22 的 logits_max_abs 解释不了 1.375 margin 的翻转)——待诊断。
  - G4 FAIL:baseline 中位 54.83 ms、fused 54.41 ms,**−0.76%**(门 −8%);CV 0.34%/0.23%(A/B 有效,非噪声)。
- **判定**:FAIL(按冻结门不采用;flag 默认关;代码保留为 opt-in)。
- **重大发现(比融合本身更重要)**:launch −960 且 **device 时间 −10.7 ms**,步时却只 −0.4 ms → GPU idle 从 ~23 ms 扩大到 ~33 ms。**步时既不由 device 工作量决定,也不由 launch 提交量决定**。I19"launch-bound→融合换步时"的因果假设被实测否定;此前文档里的"13.5 µs/launch"是 device 时间均摊(31.9ms/2402),不是 host 提交成本。图的 replay(P2)成为判决"每步固定成本到底是什么"的决定性实验。
- **决策**:转 P1.3 host-sync A/B(2 次 bool sync + 96 次 tolist 是仅剩的硬 sync 点);G2 矛盾做逐模块 bitwise 诊断。
- **证据**:`/root/p8_fusion.json`(远端),`/root/p8_fusion_ab.log`。

#### P1.1 附加诊断 — bitwise 分歧源头(H2.6)

- **问题**:G1 kernel probe bitwise 全等 vs G2 端到端翻转,矛盾待解。
- **方法**:`tests/p8_rmsnorm_bitwise_diag.py`,113 个 RMSNorm 模块逐 call 输出比对(prompt 6)。
- **实际**:每次 call 0–1 元素差 1 bf16 ulp(首分歧 `layers.1.input_layernorm`,[128,1024],max_abs=4.88e-4);3,573/3,616 call 有微差;**本次运行 tokens_match=True(A/B 中的翻转未复现)**。原因:eager `mean` 的归约切分依赖输入形状(probe 的 37/64/17 行恰好与 Triton 一致,真实 128 行不一致),1-ulp 差异经层放大。
- **结论**:G2 FAIL 的机理明确——任何非 bitwise 等价的 RMSNorm 重实现都会带来 I4 同级漂移;融合 kernel 本身数值健康(L1 4.9e-4 ≪ 2e-3),按翻转率门判不采用。
- **证据**:`/root/p8_rmsnorm_bitwise_diag.json`、`/root/p8_rmsnorm_diag.log`。

### H2.7– P1.3 host-sync A/B — 冻结门(跑前写)

- **假设**:0.6B b32 混合温度时 sampler 的 2 次 `bool()` 设备读是每步硬串行点;prefill 每层 2 次 `.tolist()`(512-token prefill 96 次)拖慢 TTFT;去掉后 sample 段与 prefill 步时可测改善。
- **唯一变量**:`set_host_sync_fix`(on/off,同进程交替,env `NANOVLLM_HOST_SYNC_FIX`);其余栈不动。
- **冻结门**:
  - T1:b1 与 b32-greedy 的 token 流 on/off 全等。
  - T2:b32-mixed 的 sample 段中位数 on ≤ 0.5×off,CV≤5%(两组)。
  - T3:prefill-512 步时 on < off(0.6B 上仅报告,−≥50% 门是 30B L2 的事)。
  - T4:b1/b32-greedy 步时差仅报告。
- **判定**:T1+T2 过 → 代码保留为默认开(on),进 tiny 回归;T2 不过但 T1 过 → 保留代码但结论记"0.6B 不可测,30B L2 复核"。
- **证据**:`/root/p8_sync_fix.json`、`/root/p8_sync_ab.log`。

#### P1.3 host-sync A/B — 实际结果(H4.7)

- **实际**(交错 5+5,同进程):
  - T1 PASS:b1(32 步)与 b32-greedy(16 步)token 流 on/off 全等(注:第一版脚本 b1 用了不同 prompt、b32 用 seq_id 键控 dict 比较,均为测试 bug,修正后 PASS)。
  - T2(b32-mixed sample 段):1.75 → 1.51 ms,−13.5%;0.5× 门不过——该门按 30B b32 sample 36.6 ms 设计,0.6B 的 sample 段被 argmax+tolist 主导,0.6B 上不可判,30B L2 复核。
  - T3(prefill-512 步时):run1 62.8→55.5 ms(**−11.6%**),run2 62.3→58.9 ms(−5.4%);两组 wall 交错clean分离,CV ≤1%。
  - T4:b1 步时 52.95→52.73(−0.4%)、b32-greedy 60.90→60.79(−0.2%)——再次确认 decode 步时对 host-sync 摘除不敏感。
- **判定**:PASS(T1+T3);代码默认 ON 采纳。prefill 的 96 次 tolist 是真实可测开销(~7 ms/512-token prefill step);decode 侧的 2 次 bool 仅 ~0.2-0.4 ms。
- **决策**:host-sync 修复保持默认;A 项验收的"−≥8%"主张挂到 30B L2 的 prefill 实测。继续 P2。
- **证据**:`/root/p8_sync_fix.json`、`/root/p8_sync_ab4.log`。

### H4.8– P2 图 replay 根因 — 冻结门(跑前写)

- **假设**(架构书 §2.7):replay 时 padding 行 slot_mapping=0,整桶执行覆写 KV slot 0(block 0 常属活跃请求)→ 首 replay 漂移。
- **修复**(已实现,`model_runner.py`):allocate_kv_cache 多留 1 个 scratch block(BlockManager 不可见);replay 前 `slot_mapping.fill_(scratch_slot)`、`block_tables.fill_(scratch_block_id)`,真实行再覆盖前缀。
- **唯一变量**:scratch-slot 修复(tiny/8B 图 vs eager)。
- **冻结门**:
  - GA(slot0 回读):replay-only decode 步间,KV slot 0(全层 K/V)bitwise 不变。
  - GB(5/5):tiny 连续 5 次 graph generate 与 eager 参考 token 全等(修复前 0/5)。
  - GC(共享前缀):共享 300-token 前缀的两请求 graph == eager(block 0 被共享的场景)。
  - GD:捕获失败与 replay 漂移分开计数。
  - 顺序:tiny 5/5 → 8B 5/5 → 才允许 30B。
- **30B 图门来源审计(跑前写死,双轨)**:
  - 硬门:graph 与 eager 逐 token 全等(b1/b8/b32);吞吐 CV ≤5%;图显存增量 ≤2 GiB。
  - 性能门(双轨,b1):**≥+15% 转正**;+8~15% 记录"正确性通过、性能待产品决策"不改默认;<+8% 判"图在本栈不划算"。
  - b8 ≥+5%、b32/b64 ≥0.98 仅报告,不作唯一否决。
  - 注:该 +15% 门定于 bmm_dense 采用之前;现在 eager 已拿走 decode 大头,图增量必然变小,故用双轨而非单一否决。
- **证据**:`/root/p8_graph_padfix.json`(tiny/8B)、`/root/p8_graph_30b.json`(L1)、`benchmarks/P8-graph.md`。

#### P2.2 tiny/8B 验证 — 实际结果(H6.2–H7.0)

- **0.6B(paged,dense)**:GB PASS(graph 5/5 与 eager 逐 token 全等,生产 attention 后端);GC PASS(共享 300-token 前缀两请求全等);GD=0(捕获零失败);GA 未判定(见下)。
- **8B(paged,dense)**:**GA+GB+GC+GD 全 PASS,overall PASS**(5/5 token 全等、slot0 回读干净、共享前缀全等、捕获零失败)。
- **tiny MoE(sdpa,head_dim=16 不可用 paged)**:GA/GC PASS、5/5 自洽,但 GB FAIL——机制已定位:图捕获烘焙 capture 时的 kv 窗口形状(256),eager 用运行时窗口,SDPA 归约序不同 → I4 级 1-ulp 漂移;**paged kernel 的循环边界在 device 上由 context_lens 决定,不受烘焙形状影响**,故生产路径(paged)上 graph==eager bitwise 成立(0.6B/8B 实证)。
- **GA 波折(如实记录)**:slot0 回读三次 FAIL,根因是**测量基础设施**:torch.musa 对 `kv_cache[:, :, blk, 0]` 跨块步长视图的 D2H 拷贝返回与 device 侧逐层扫描不一致的数据(full-slice NaN、K/V 分读干净、`.item()` 扫描干净、contiguous() 后仍 NaN)——平台 bug 嫌疑,已加 workaround 与说明。GA 的实质(活跃 KV slot 不被 replay 覆写)由 GB 独立证明:若覆写,token 必漂移。记入 ISSUES(新条目)。
- **判定**:P2.2 PASS(padding-slot 根因确认并修复;paged 路径 graph==eager)。**允许 30B L1**。
- **证据**:`/root/p8_graph_padfix.json`(0.6B)、`/root/p8_graph_padfix_8b.log`+json(8B)、`/root/p8_graph_padfix_tiny` 首轮 json(tiny,sdpa 解释性 FAIL)。

#### P2.4 30B L1 — 第一次尝试楔死(记录)与重启(H9.3)

- **现象**:捕获 6 桶成功(显存 allocated 36.08 GiB 稳定、reserved +0.34 GiB ≤2 GiB 门 ✓、KV explicit 640/640 两 rank 一致 ✓);warmup replay 阶段进程 CPU 0%、**双卡 100% util 对称自旋**。
- **根因(A/B 脚本,非引擎)**:`drive_batch` 只改了 rank0 的 `runner.enforce_eager`;rank1(独立进程)仍是 init 时的 False 走 replay → 同一步两 rank 集体通信路径错配(eager collective vs 图内 collective)→ MCCL 双向等待死锁自旋。
- **修正**:改用 `runner.call("set_enforce_eager", ...)`(shm 命令通道 + barrier,两 rank 同步切换)。已按 SOP 按 PID kill(rank0 被 pkill 命中,rank1 spawn 子进程按 PID 补杀)→ 清 shm → 双卡 0 MiB → 重跑。
- **SOP 增补**:任何 eager/graph 运行时切换必须走 `model_runner.call`(两 rank),绝不直接写本地属性。

### H9.5– P3 (L2) — 冻结门(跑前写)

- **P3.1 真实负载统计**:32 条中英 chat prompt(512 token)、decode 64;fold 前 owned/folded 计数 + 全局 counts;输出 Gini、rank imbalance(folding 修正口径)、S_max 观察(shrink 路径 (tokens, max_load, n_owned) 记录)。判定:**如果 max_load 中位数 ≥ 2× 均值,容量收缩收益打折;否则收益成立**。
- **P3.2 容量收缩 A/B**(唯一变量 = `_CAPACITY_SHRINK`):
  - 数值门:两模式各 3 次自确定(PASS/FAIL);与 padded 的 token 一致率仅报告(bmm tiling 序变化,预期非全等;30B 上 token agreement 记录)。
  - 性能门:b32 与 b64 decode 步时中位数 **−≥10%**(冻结门取步时;MoE 段 −20% 由 S_max 推算佐证),交错 5+5、CV ≤5%,否则 INVALID。
  - 显存:x_pad 峰值显著低于现状(观察)。
- **P1.3-30B(prefill host-sync)**:512-token prefill 步时 on/off 交错 5+5;**门:on < off 且 delta ≤ −5%**(0.6B 实测 −11.6%;30B 若 ≥5% 即可作为 A 项验收主证据)。
- **证据**:`/root/p8_moe_30b.json`。

#### P2.4 30B L1 — 第二次尝试:replay 楔死复现 → P2 结案(H10.1)

- **现象**(切换已修复为 `runner.call("set_enforce_eager")` 双 rank 同步):eager b1 trial 正常(142.06 ms/decode step),**首个 graph-mode trial 再次楔死**——双卡 100% util、host CPU 0%。
- **判决**:**30B TP2 的首次 replay 楔死是独立于 padding-slot 的第二个故障**。证据:①0.6B/8B(TP1,图内无 collective)padding 修复后 replay 完全正常且 bitwise 全等;②楔死时双卡对称自旋,符合图内 96 次 MCCL all-reduce 的 rendezvous 失败特征;③与 P6 nightly 记录的"30B 捕获成功、首次 replay 楔死"一致。padding-slot 修复解决的是"漂移"类故障(TP1 已实证);TP2 wedge 属 MCCL-in-graph 问题,本轮不做(4h 硬停已过)。
- **按双轨门判决**:硬门未完成(token 全等无法测,流程楔死)→ **30B 图维持否决:默认保持 eager**;性能门不适用。
- **本轮图 Track 净产出**:①padding-slot 根因确认+修复+TP1 全链验证(tiny/0.6B/8B);②sdpa 图烘焙形状的数值限制定位;③TP2 wedge 与漂移分离为两个故障;④运行时切换 SOP 增补。
- **复活条件**:MCCL-in-graph 的 replay 支持性探针(参照 tests/p6_mccl_capture_multi.py 的多桶结构)+ 双 rank 同步 replay 协议;或分段图(attention/linear 进图、MoE 与 collective eager)。
- **证据**:`/root/p8_graph_30b.log`、`/root/p8_graph_30b2.log`(楔死现场)。

### H10.1–H11.2 P3 (L2) — 实际结果

- **P3.1 真实负载统计**(32 条中英 chat,512-token prefill + 64 decode):routing 总 slot 6.87M;**waste(folded 无效行)47.1%**(与 §2.2 预测 ~50% 吻合);Gini 0.296;**rank imbalance 均值 15.8%**(首层 rank0 63,654 vs rank1 79,450)——真实负载的跨 rank 不均衡远超 synthetic 的 1.7–5.2%,profiled placement 值得下轮第一优先;S_max 观察:b32 中位 25 / 最大 32(最坏 256),收缩机制按设计工作。
- **P3.2 容量收缩 A/B 判决**:
  - 数值门 PASS:b32/b64 与 padded **token 全等**(行级 K 维归约与 M 位置无关,bitwise 一致,优于预期的软门)。
  - 性能门 **FAIL**:b32 308.47→303.91 ms(**−1.5%**)、b64 439.41→439.89(+0.1%),门 −10%;CV 0.2–1.0%(A/B 有效,非噪声)。
  - **判定:不采用(默认关);设计保留。** S 从 256→25(bmm 行数 ~10×)却不动步时 → 与 P1 结论一致:30B decode 步时既不由 MoE bmm 工作量、也不由 launch/sync 决定。
- **P1.3-30B prefill host-sync**:424.9→409.6 ms(**−3.6%**,on/off 区间干净分离),未达 −5% 门;0.6B −11.6% 仍为本项最佳证据。
- **P3.3 placement A/B**:未跑(需换 placement 重载 27 GiB 权重 ≈ 2×7 min;时间让给验收 D/E)。**结论:profiled map 生成器与统计就绪,真实 imbalance 15.8% 是动机,下轮先跑**。
- **证据**:`/root/p8_moe_30b.json`(514 s)、`/root/p8_moe_30b.log`。

### H12– H13.5 P4 L3 长上下文重测 — 实际结果

- **1k 三档(chunk 256,unique,v3 客户端)**:有效门全过(prompt p50 1086/1154/1154、128×3 全 DONE、rank1 存活、零 preempt、错误桶全 none)。
  - TTFT p50 = **3.31 / 4.05 / 5.04 s**(上轮 3.84 / 8.28 / 10.22;75%/100% 档 **−51%/−51%**);TPOT p50 = 0.368 / 0.624 / 0.876 s。
  - **服务端 queue p50/p95 ≈ 0.05–0.17 ms** → TTFT 无排队成分,全是 1k prefill 计算;I23 mixed-split + I26/HOL 修复把高负载档 TTFT 砍半,但**改进门(50% 档 ≤2 s)未达(3.31 s)→ I26/I23 部分解决**;调度层已无免费收益,下一层是 prefill 计算路径。
- **2k 单档(并发 64)**:128/128、零 preempt、rank1 存活、GPU 峰值 46.7 GiB;prompt p50 2193 越带上限 0.8% → 判"边界有效",核心观测(preempt=0)有效。
- **chunk 256 vs 2048 对照**:未跑(需重启服务器重载 30B;余额紧张,记录为下轮项)。
- **判定**:D 验收 = 部分过(有效性 PASS、改进门 FAIL、2k 边界)。
- **证据**:`/root/p8_long_context_c256.json`、`/root/p8_long_context_2k.json`、`/root/p8_l3_server_256.log`、`/root/p8_l3_metrics_256.jsonl`。

### H13.5– P5 最小并发曲线(L4,补救性最小集)

- 服务器(30B,chunk 256)已载,白捡一次:`--mode concurrency --fractions 1,4,16,32,64 --requests 32 --max-tokens 64 --prompt-words 128`;Poisson ×3 CI 未做(记录)。
