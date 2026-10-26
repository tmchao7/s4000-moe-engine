# HANDOFF-P6b — 15.6 GiB 定案、bmm_dense 落地、图 replay 的间歇性挂死

> 写给接手 12 小时夜间迭代的 agent。**先读 §0 / §1 / §4，再决定做什么。**
> 本文件只写本轮实测出来的事实；标【已实测】的可以直接引用，标【待验证】的必须自己复现。
> 上一份交接 `docs/HANDOFF-P6-next.md` 的 T1 假设（厂商 workspace）**是错的**，见 §2.1。

---

## 0. TL;DR（30 秒）

1. **P6 的 15.6 GiB OOM 不是厂商 workspace**，而是我们自己的惰性权重副本：`gate_up` 384 MiB + `down` 192 MiB = 576 MiB/层，48 层要 27 GiB，在第 25 层 OOM。【已实测】
2. **零拷贝厂商 MoE 可行**：把 native `[E,in,out]` 专家权重以转置**视图**喂给 vLLM Triton kernel，数值正确、显存 +0.0 MiB、可捕获、replay 一致。【已实测】但默认配置下比 `grouped_mm` 慢 2.3–6.3×，**不能当默认**。
3. **本轮真正的性能赢家是 `bmm_dense`**：容量填充的稠密 `torch.bmm`，原生权重零拷贝，在 decode 形状上比 `grouped_mm` 快 **1.3–1.95×**，且**可捕获**。【已实测】已集成为 `decode_moe_backend="bmm_dense"`（只用于 decode，prefill 仍 `grouped_mm`）。
4. `grouped_mm` 的 route/pack 已改成**无主机同步**（去掉布尔索引、`bincount`、`.any()`），eager 下快 1.14–1.51×。【已实测】
5. 找到并修掉 **shm 命令通道竞态**：rank0 会覆盖 rank1 还没读完的命令，表现为 `pickle data was truncated` / rank1 执行错步；这**很可能**是历史 b32 不可复现和 b64 崩溃的根因。【已实测截断错误；30B 上的稳定性待验证】
6. **图捕获本身成功**：30B 的 6 个桶（1/2/4/8/16/32）全部捕获，两 rank 一致，显存增量 0.34 GiB。【已实测】
7. **图 replay 在模型规模上间歇性挂死**（8B TP2 / 30B）：捕获后第一次或第二次 `generate` 即触发 10 分钟 watchdog `launch timed out and was terminated`。tiny（TP1/TP2）不挂；纯 MCCL 探针（96 次 all-reduce、eager 后再 replay）不挂。**这是本轮唯一未解的技术硬问题**，见 §4.1。【已实测症状；根因未定】

**夜间迭代的正确优先级（不要歪）**：
**A. 先把 eager 的 `bmm_dense + triton_paged + shm 修复` 在 30B 上验证并落地（容量故事就在这里）→ B. 用 D2 验证 b32/b64 是否被 shm 修复治好 → C. 在线容量复测 → D. 最后才在图 replay 上做 2–3 小时的时间盒诊断。** 图现在不是主路径。

### 0.1 本轮状态板

| 项 | 状态 | 证据 |
|---|---|---|
| T1 OOM 归因 | ✅ 定案：惰性权重副本 576 MiB/层，非 workspace | `/root/p6_moe_memory.json` |
| 零拷贝厂商 MoE | ✅ 可行（数值/显存/捕获） | `/root/p6_vendor_zero_copy.json` |
| 厂商 MoE 性能 | ❌ 默认慢 2.3–6.3×；mp22 数值错 | `/root/p6_moe_microbench2.json` |
| `bmm_dense` decode 后端 | ✅ 1.3–1.95×，可捕获，已集成 | 同上 + `tests/p6_bmm_dense_tiny.py` |
| route/pack 无同步化 | ✅ eager 快 1.14–1.51× | 同上 |
| MCCL 延迟/确定性/图捕获 | ✅ 0.08 ms/call；bitwise 确定；96 次可捕获 | `/root/p6_mccl*.json` |
| shm 命令通道竞态 | ✅ 已修（每条命令 barrier） | 8B A/B 的 `pickle truncated` |
| 30B 图捕获 | ✅ 6 桶，+0.34 GiB | `/root/p5_graph_ab.log` |
| 30B 图 replay | ❌ 间歇性挂死（watchdog） | 同上 + `p5_graph_ab_8b*.log` |
| 30B eager 新栈端到端 | ⏳ 未跑（下一优先级） | — |
| b32/b64 稳定性（shm 修复后） | ⏳ 未验证 | — |
| 在线容量 v2 复测 | ⏳ 未跑 | — |

---

## 1. 服务器与代码状态

```bash
ssh -p 28036 root@connect.sha1.seetacloud.com      # 已配免密
# 代码（服务器）：/root/moe-server-s4000   ← 注意：这里的 .git 是空的/伪仓库，不要依赖
# 本地：/Users/tmchao7/repos/02_personal/moe-server-s4000
# 分支 feat/musa-port，HEAD ecc87b2，工作区有本轮**未提交**改动（§3）
```

- 模型：`/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8`（加载 ~13–15 分钟，加载期反量化为 bf16，每卡 ~28.6 GiB）
- tiny：`/tmp/tiny_moe`（fp32，2 层，head_dim=16 → **不能用 triton_paged**）
- 8B：`/root/models/Qwen3-8B`（TP2 可用，~1 分钟加载）
- 图 A/B 使用：`--num-kvcache-blocks 32 --gpu-memory-utilization 0.72`，权重后 free ~16.3 GiB
- 服务器 SSH 在宿主高 load 时会**频繁断连**；长任务用 `setsid nohup` + 短连接轮询，rsync 失败就重试。

**本轮收手时已清理**：卡死的 8B 进程已 `kill -9`，两卡回到 0 MiB、无残留进程，`/dev/shm/nanovllm*` 已清。接手时仍建议按下面复核一遍（服务器 SSH 在高 load 时会断，长任务务必 `setsid nohup`）：

```bash
mthreads-gmi | grep -E "^[01] "
pgrep -fa "^python3 tests/p6_graph_min|[s]pawn_main"
# 用 PID 直接 kill -9，不要用 pkill -f（见 §7）
rm -f /dev/shm/nanovllm*
```

---

## 2. 本轮定案的事实

### 2.1 T1：那 15.6 GiB 到底是什么【已实测，推翻旧假设】

证据：`tests/p6_moe_memory_probe.py` → 服务器 `/root/p6_moe_memory.json`、`.log`。

| 实验 | 结果 |
|---|---|
| 同一层权重，连续 48 次 `fused_experts(1/8/32/64 token)` | 每次显存增量 **0.0 MiB**（64 token 时 0.2 MiB） |
| 28.6 GiB filler + 0.38 GiB KV，然后逐层做 `transpose(1,2).contiguous()` | 每层精确 **384.0 + 192.0 MiB = 576 MiB**；第 25 层在「分配 192 MiB」时失败，已分配 43.98 GiB、剩余 104 MiB |

失败分配 `384 MiB = 64×2048×1536×2`，正好是一层 `gate_up_proj.transpose(1,2).contiguous()`。48 层需要 27 GiB，与 30B 现场（44.13 GiB 时失败、只剩 282 MiB）完全吻合。

**结论**：旧 `HANDOFF-P6-next.md` 的 T1（workspace13/workspace2 按 16384 token 分配）是错的。厂商 Triton MoE 的 workspace **不随层增长**。真正的阻塞是 `qwen3_moe.py` 里为满足 vLLM `stride(-1)==1` 而做的惰性转置副本。

### 2.2 零拷贝厂商 MoE【已实测】

证据：`tests/p6_vendor_zero_copy.py`（用 `python3 -O` 绕过 assert）→ `/root/p6_vendor_zero_copy.json`。

- `w1 = gate_up.transpose(1,2)`（shape `[64,1536,2048]`，stride `(3145728,1,1536)`），`w2 = down.transpose(1,2)`，直接喂 `fused_experts`。
- 1/8/32/64 token：`max_rel vs fp32` = 0.0040 / 0.0036 / 0.0051 / 0.0052；显存增量 **0.0–0.1 MiB**；捕获 + 改 ids replay：`replay_max_rel = 0.0`，`stale=false`。
- 集成方式：`nanovllm/kernels/vendor_moe.py` 用 `inspect.getsource` 在**内存里**重绑定 `fused_experts_impl`，只删掉 `w1/w2.stride(-1)==1` 两行 assert；**没有改 site-packages**。数值由上面的探针把关。

### 2.3 MoE 后端微基准【已实测】

证据：`scripts/p6_moe_microbench.py` → `/root/p6_moe_microbench2.json`、`.log`。
形状：E=64（每 rank）、hidden=2048、inter=768、top-8/128；单层 eager ms（48 层即一步的 MoE 时间）。

| batch | grouped_mm 旧（有同步） | grouped_mm 新（无同步） | **bmm_dense** | vllm_fused 默认 | vllm_fused mp22 |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.1025 | 2.0543 | **1.2684** | 4.6946 | 0.2943（rel=1.60） |
| 2 | 3.1878 | 2.1366 | **1.2994** | 4.9588 | 0.3162（rel=1.39） |
| 4 | 3.4966 | 2.3815 | **1.3747** | 8.5906 | 0.4446（rel=1.35） |
| 8 | 4.1328 | 2.9249 | **1.9444** | 15.3871 | 0.7733（rel=1.63） |
| 16 | 5.2039 | 4.1087 | **2.3014** | 28.2180 | 1.4051（rel=1.45） |
| 32 | 6.6821 | 5.5613 | **2.8582** | 38.5472 | 1.5115（rel=341） |
| 64 | 7.8332 | 6.8716 | **5.1694** | 43.0936 | 2.2770（NaN） |

- `bmm_dense` 与 syncfree 的 `max_rel`：0.0015–0.0031（bf16 累加顺序差异，可接受；贪心 token 在 tiny 上与 loop 完全一致）。
- `bmm_dense` 的图 replay ≈ eager（graph_ms 与 eager_ms 相差 <2%），说明**收益来自 kernel 本身，不是图**。
- `vllm_fused` 默认配置慢 2.3–6.3×；vLLM 在找 `configs/E=64,N=768,device_name=MTT_S4000.json`，**该文件不存在**（autotune 未做）。
- `mp22` 快 7–15×，但**数值完全错误**（rel 1.3–341，b64 NaN）→ 永不可用，除非厂商修复。

**结论**：decode 后端选 `bmm_dense`；prefill 保持 `grouped_mm`；`vllm_fused` 只保留为零拷贝能力证明，不做默认；`mp22` 禁用。

### 2.4 MCCL【已实测】

证据：`tests/p6_mccl_determinism.py`、`tests/p6_mccl_capture_multi.py` → `/root/p6_mccl.json`、`/root/p6_mccl_capture_multi.json`、`/root/p6_mccl_eager.json`。

- 延迟：back-to-back 96 次 all-reduce，每次 **0.076–0.103 ms**，一步 96 次仅 **7.5–9.9 ms**。b1–b64、4 KiB–256 KiB 一致。
- 确定性：100 次重复，1–64 batch、4 KiB–256 KiB 全部**逐 bit 相同**。
- 图捕获：单图 96 次 all-reduce、bf16/fp32、4 KiB/128 KiB/256 B，replay `max_rel=0.0`；**捕获后再 eager 100 次再 replay 也正确**。
- 结论：MCCL **不是** 300 ms/步的来源，也**不是** b32 不可复现的直接来源（孤立测试下）。

### 2.5 每步开销（8B dense，TP1）【已实测】

证据：`tests/p6_step_overhead.py` → `/root/p6_step_overhead.json`。

| batch | eager 步 ms | graph 步 ms | 图收益 |
|---:|---:|---:|---:|
| 1 | 74.3（forward 71.6 / sample 2.4） | 63.6 | 1.17× |
| 8 | 87.0 | 74.1 | 1.17× |
| 32 | 89.5 | 87.3 | 1.03× |

- pinned buffer 每步重建总成本 **0.53 ms**（可忽略）。
- eager 的 forward 慢，部分来自 `store_kvcache` 的 `(slot_mapping<0).any()` 每层一次主机同步；已移除（§3）。
- **注意**：这个 1.17× 是在移除 route/kv 同步**之前**测的。移除后 eager 更快，图的增量收益可能更小。不要用旧数字承诺图收益。

### 2.6 图捕获：成功；图 replay：间歇性挂死【已实测症状，根因未定】

**捕获成功**（30B，`/root/p5_graph_ab.log`）：
- 桶 `[1,2,4,8,16,32]`，两 rank 全部完成；`decode_moe_backend=bmm_dense`、`attention_backend=triton_paged`。
- 图显存增量 `reserved 0.34 GiB`（门限 2 GiB）；KV 32 blocks = 0.38 GiB；捕获前 free 16.3 GiB。
- 捕获稳定性：加 rank barrier + 设备同步后，tiny TP2 连续 3/3 成功。

**Replay 挂死**：
- 30B：捕获后进入 `p5_graph_ab.py` 的 `verify graph_candidate`，约 10 分钟后 watchdog：`RuntimeError: MUSA error: the launch timed out and was terminated`（栈在 sampler；rank1 出现大量 `MUSAGraph.cpp ... (function reset)`）。
- 8B TP2：`p5_graph_ab.py` 同样卡在 `verify graph_candidate`；日志 `/root/p5_graph_ab_8b.log`、`/root/p5_graph_ab_8b2.log`。
- 8B TP2 单次 generate 的 `p6_graph_min.py` **曾经成功**（`/root/p6_graph_min_tp2.log`，tokens 正常）；后来 `--generates 3 --clear-prefix-cache` 卡在 `generate 1/3`（`/root/p6_8b_3gen.log`）。
- tiny TP1/TP2 的 graph → eager → graph **不挂**（`/root/p6_replay_after_eager*.log`），但两次 graph 的 token 不一致（tiny 用 SDPA，kv_len 冻结；不可作为 30B 的结论）。
- 纯 MCCL 的「捕获 → eager 100 次 → replay」**不挂**。

**因此：不是“图不可捕获”，也不是 MCCL 能力问题；是模型规模下、图 replay 与 eager/其他资源交互的间歇性挂死。** 见 §4.1 的二分计划。

### 2.7 b32 不可复现 / b64 崩溃的新解释【待 30B 验证】

`nanovllm/engine/model_runner.py` 的 shm 命令通道是**单缓冲、无 ack**：

```python
def write_shm(...):  # rank0 写 [4字节长度][pickle]
def read_shm(...):   # rank1 读
```

rank0 在本地 `run` 返回后立刻写下一步；rank1 可能还在 unpickle 上一步。8B A/B 已实测到 `_pickle.UnpicklingError: pickle data was truncated`。rank1 执行截断/错位命令，足以解释：
- b32 两个 attention 后端都“不可复现”；
- b64 第三次重复 `MUSA unknown error` / rank1 退出；
- 图 replay 与 rank1 的集合通信错位（虽然纯 MCCL 探针不挂，但模型图更复杂）。

**修复**：`ModelRunner.call()` 在每条命令（除 `exit`）后 `dist.barrier()`（~0.08 ms）。【已实测截断消失；30B 稳定性待验证】

---

## 3. 代码改动清单（工作区未提交）

**修改**
- `nanovllm/models/qwen3_moe.py`
  - 新增 `bmm_dense` 后端 `_forward_bmm_dense`：`S = tokens*top_k` 精确容量；`argsort` + `index_add_` 算 offsets；`x_pad[sorted_ids, ranks] = x_slots`；两次 `torch.bmm`（原生 `[E,in,out]`）；确定性 scatter + `sum(1)`。**只用于 decode**。
  - decode/prefill 按 `get_context().is_prefill` 选 `decode_moe_backend` / `moe_backend`（**这是关键语义**：不是“图内后端”，eager decode 也用）。
  - `grouped_mm` route/pack 改为**无同步**：非本地专家折叠到 expert 0 + 权重置零；`counts.index_add_` 替代 `bincount`；去掉所有布尔索引和 `bool(valid.any())`。
  - `_forward_vllm_fused` 改为零拷贝视图；删除 `_vllm_weights` 惰性副本；新增 `global_to_local_i32` buffer。
- `nanovllm/kernels/vendor_moe.py`（新）：零拷贝厂商适配层；内存内重绑定 `fused_experts_impl`（只删两行 layout assert）；`use_mp22` 开关（默认 False）。
- `nanovllm/config.py`：`bmm_dense` 加入 `moe_backend` / `decode_moe_backend` Literal 与断言；`decode_moe_backend` 注释改为“decode 步的后端（eager 或图）”。
- `nanovllm/layers/attention.py`：`store_kvcache` **始终**直接 `flat_k[slot_mapping]=key`。**布尔掩码 `slot_mapping[valid]` 不是 capture-safe**（mask index 需要 nonzero 的主机同步）；`-1` 只存在于图静态缓冲尾部，捕获的视图只覆盖前 N 行，replay 不会读。同时去掉了每层一次 `.any()` 主机同步。
- `nanovllm/engine/model_runner.py`
  - `set_moe_backend(backend, decode_backend=None)`：同时设置两个后端。
  - `capture_cudagraph`：捕获用**真实上下文**（`slot_mapping=arange(max_bs)`、`context_lens=min(block_size,max_model_len)`），不再是全零（全零会让 SDPA 拿到全 False mask，且在 capture_begin 的 synchronize 处留下 `operation not permitted`）。每个桶前后加 `dist.barrier()` + `acc.synchronize()`（MCCL 捕获需要两 rank 对齐；tiny TP2 3/3 稳定）。
  - `run_model` replay：padding 行的 `slot_mapping` 填 **0**（不是 -1，-1 会写 KV 末尾），`context_lens` 填 1（避免 0/0 的 NaN padding 行）。
  - `call()`：每条命令后 `dist.barrier()`（`exit` 除外）——修 shm 竞态。
- `scripts/p5_graph_ab.py`：config-major 顺序（graph → eager_candidate → eager_default）；`set_moe_backend(prefill, decode)`；进度打印。

**新增探针/脚本**
- `tests/p6_step_overhead.py`（8B 每步拆分 / 图 vs eager）
- `tests/p6_moe_memory_probe.py`（T1）
- `tests/p6_vendor_zero_copy.py`（零拷贝）
- `tests/p6_mccl_determinism.py`、`tests/p6_mccl_capture_multi.py`
- `tests/p6_bmm_dense_tiny.py`（tiny 六配置 + graph vs eager）
- `tests/p6_graph_min.py`（最小图捕获；`--generates` 可测多次 replay）
- `tests/p6_graph_replay_after_eager.py`（graph→eager→graph）
- `scripts/p6_moe_microbench.py`（三后端 + mp22 + bmm_dense 微基准）

**未改默认值**：`enforce_eager=True`、`attention_backend=sdpa`、`moe_backend=grouped_mm`、`decode_moe_backend=None`。不要在门没过时改。

---

## 4. 未解问题与夜间迭代路线

### 4.1 P0：图 replay 间歇性挂死（唯一技术硬问题，时间盒 2–3 小时）

**先建立可重复的最小复现**（不要直接上 30B，每次 15 分钟）：

```bash
# 8B，TP2，triton_paged，捕获后单次 generate；连跑 5 次统计成功率
for i in 1 2 3 4 5; do
  mthreads-gmi | grep -E "^[01] "; rm -f /dev/shm/nanovllm*
  PYTHONPATH=/root/moe-server-s4000 python3 tests/p6_graph_min.py \
    --model /root/models/Qwen3-8B --dense --tp 2 \
    --attention-backend triton_paged --generates 1 > /root/gmin_$i.log 2>&1
  echo "RUN $i EXIT=$?"
done
```

如果 5/5 成功，再跑 `--generates 3 --clear-prefix-cache`（当前会卡在 1/3）。
如果单次就间歇性失败，按下面顺序二分（每步只改一个变量，**每次改完连跑 3 次**）：

1. **rank 对齐**：在 `run_model` 的 `graph.replay()` 前后各加 `dist.barrier()`（在 barrier 前后不要有主机同步进图）；或 `torch.musa.synchronize()` 后再 replay。怀疑 rank0 的 replay 与 rank1 的 eager/其他步错位。
2. **显存池**：`capture_cudagraph` 每个桶用**独立 pool**（`self.graph_pool=None` 每桶），排除跨桶 pool 复用与 MCCL 捕获指针的冲突。8B 单独捕获成功、A/B 卡死，这个假设值得优先试。
3. **padding 行**：`run_model` 把 padding 行的 `slot_mapping` 设为**互不相同的有效槽**（例如 `bs + i`，确保 < num_kvcache_blocks*block_size），而不是都填 0；排除同一槽的并发写/竞争。
4. **sampler 主机同步**：`nanovllm/layers/sampler.py` 的 `bool(greedy.all())`/`bool(greedy.any())` 每步一次同步；图 replay 后立即在 sampler 同步。试把 greedy 分支改成纯 `torch.argmax`（无 `bool`），或把采样移出图后的关键路径。
5. **capture_error_mode**：试 `relaxed`；确认 torch_musa 2.11 的 `MUSAGraph` 对 MCCL 的 replay 是否有已知限制。
6. **只捕非 MoE 段**（breakable）：如果以上都不行，退到 SGLang 式分段捕获（attention/linear/allreduce 进图，MoE eager）。这是最后手段，工作量大。

**判定**：如果 3 小时内不能把 8B 的单次 generate 做到 5/5 稳定，**停止图方向**，把 §4.2/§4.3 做完。图捕获的算子级结论（§2.6）本身已经是有价值的 P6 结果；不要在夜间把全部时间烧在一个厂商 runtime 的挂死上。

**若图修好**，再跑 30B A/B（一条命令，见下），门不变：

```bash
cd /root/moe-server-s4000
mthreads-gmi | grep -E "^[01] "; rm -f /dev/shm/nanovllm*
PYTHONPATH=/root/moe-server-s4000 PYTHONUNBUFFERED=1 setsid nohup \
  python3 scripts/p5_graph_ab.py --batches 1,2,4,8,16,32 --repeats 3 \
  --gpu-memory-utilization 0.72 --num-kvcache-blocks 32 \
  --decode-moe-backend bmm_dense --prefill-moe-backend grouped_mm \
  --attention-backend triton_paged --output /root/p5_graph_ab.json \
  > /root/p5_graph_ab.log 2>&1 < /dev/null &
python3 scripts/p5_decide.py --mode graph --input /root/p5_graph_ab.json \
  --output benchmarks/p5-graph-ab-decision.json --markdown /root/p5_graph_ab.md
```

门（`scripts/p5_decide.py::GRAPH_GATES`，**禁止改**）：token 全等、b1 ≥ +15%、b8 ≥ +5%、b32/b64 ≥ 0.98、CV ≤ 5%、显存增量 ≤ 2 GiB。

### 4.2 P1：在 30B 上验证 eager 新栈（这是容量故事的主路径）

目标：`prefill=grouped_mm` + `decode=bmm_dense` + `attention=triton_paged`，eager，无图。预期 MoE 单层快 1.3–1.95×，整步快 15–30%。

两种做法（选一）：
1. 给 `scripts/p2_benchmark.py` 加 `--decode-moe-backend`（它已有 `--moe-backends`/`--attention-backends`），跑 schema-v2 矩阵 b1..b32，与 `benchmarks/P5-final-matrix.md` 同口径对比。
2. 或写一个单次加载的 A/B：`set_moe_backend("grouped_mm","grouped_mm")` → 测；`set_moe_backend("grouped_mm","bmm_dense")` → 测；attention 同理。`call` 已带 barrier，切换安全。

验收：
- 吞吐/TPOT 在 b16/b32 达到 `throughput_ratio ≥ 1.08`、CV ≤ 5%（沿用 `MOE_GATES` 的精神；不要改 `p5_decide.py`，可在新报告里写清口径）。
- token 与冻结参考 `benchmarks/p2-reference-8tok.json` 的对比要单独报告（`bmm_dense` 累加顺序不同，**不保证逐 token 全等**；tiny 上六配置全等）。
- 显存：bmm_dense 的临时 `x_pad` 在 b64 约 134 MB/层（分配器复用），图池不在 eager 路径。

### 4.3 P1：D2 稳定性（shm 修复是否治好 b32/b64）

```bash
PYTHONPATH=/root/moe-server-s4000 python3 tests/p5_determinism_b32.py   # A/B/C/D 四组
PYTHONPATH=/root/moe-server-s4000 python3 tests/p5_d2_stability.py --repeats 10 --bisect
```
- 重点看：`grouped_mm` b32 是否稳定；`loop` 是否稳定；zero-KV 是否稳定；b16 对照；b64 三次是否不再崩。
- 若稳定：说明 shm 竞态是根因，可把 `triton_paged` 的 attention 门按“candidate 自身确定 + ≥99% token 一致”重跑 `scripts/p5_decide.py --mode attention`。
- 若仍不稳定：按 `tests/p5_determinism_b32.py` 的结论继续二分（MCCL 已排除；重点 grouped_mm/argsort/SDPA mask）。

### 4.4 P2：在线容量 v2 复测

采用通过门的栈（`decode_moe_backend=bmm_dense`，attention 视 4.3 结果），重跑 `scripts/benchmark_online.py` 的 50/75/90/100% 四档，参数与 `benchmarks/P5-serving-v2.md` 完全一致。写 `benchmarks/P6-serving-v3.md`，同表对比 v2。
- 当前 v2：50% 档 100% SLO（TPOT p95 0.455 s），75% 档 26.6%，拐点在 50→75 之间。
- 目标：把 100% SLO 的负载档推到 ≥75%。
- API server 需要加 `--decode-moe-backend` 透传（`nanovllm/entrypoints/openai/api_server.py` 目前只有 `--moe-backend` / `--attention-backend`）。

### 4.5 P3：默认值与文档

- 默认值只在门通过时改：`attention_backend`、`decode_moe_backend`；图没修好就保持 `enforce_eager=True`。
- 更新 `docs/P6-graph-capture.md`：加“捕获成功、replay 间歇挂死”的实测与复现；旧文只写了“能捕获”。
- 更新 `docs/HANDOFF.md`：L54/L154 的“P6 已判死”已过时；加载台账补本轮。
- 更新 `benchmarks/README.md`；新增 `benchmarks/P6-moe-backends.md`（§2.3 表）和 `benchmarks/P6-graph-replay-blocked.md`（§2.6）。
- 所有证据 JSON 建议 scp 到本地 `benchmarks/data/p6b/`（rsync 默认排除 `benchmarks/data`）。

---

## 5. 明确不要做的事（路子不要歪）

1. **不要把 `vllm_fused` 设成默认**：默认配置在 decode 慢 2.3–6.3×；S4000 autotune 配置不存在。除非你愿意做完整的 autotune（`VLLM_TUNED_CONFIG_FOLDER` + `E=64,N=768,device_name=MTT_S4000.json`），否则它只是零拷贝能力的证明。
2. **不要用 mp22**：数值错误（rel 1.3–341，b64 NaN）。
3. **不要全局设 `moe_backend=bmm_dense`**：prefill 的 `x_pad` 是 `[E, tokens*top_k, H]`，4096 token 时每层 ~8.6 GB，直接 OOM。必须用 `decode_moe_backend`。
4. **不要改 `scripts/p5_decide.py` 的阈值**，不要“看完结果再改门”。
5. **不要改 site-packages**：零拷贝适配在 `nanovllm/kernels/vendor_moe.py` 内存里完成。
6. **不要 `git push`**。
7. **一次只跑一个 GPU 任务**；探针会污染 benchmark 数字。
8. **不要用 `pkill -f` / 未锚定的 `pgrep -f`**：命令字符串里出现脚本名会匹配到执行命令的 bash 自己，把自己杀掉（本轮踩了两次）。用 `pgrep -f "^python3 scripts/p5_graph_ab"` / `"[s]pawn_main"`，或先 `pgrep -fa` 看 PID 再 `kill -9`。
9. **失败的 capture 会污染进程的流**：每个 case 独立进程（`tests/p6_graph_probe.py` 的协议）。
10. **不要用 tiny 的 token 一致性当图门**：tiny 是 fp32/head_dim=16，图里只能用 SDPA，kv_len 冻结，replay token 本来就会变。

---

## 6. 本轮新增踩坑（都已实际遇到）

| 坑 | 现象 | 处理 |
|---|---|---|
| shm 单缓冲无 ack | `_pickle.UnpicklingError: pickle data was truncated`；rank1 执行错步 | `call()` 每条命令后 barrier（`exit` 除外） |
| `store_kvcache` 布尔掩码 | 捕获时 `operation not permitted`（bool mask 内部 nonzero 需主机同步） | 始终直接整数 index_put |
| 捕获上下文全零 | SDPA 全 False mask，capture_begin 的 synchronize 报 `operation not permitted` | 捕获用真实 context_lens/slot_mapping |
| TP2 捕获 rank 不对齐 | 间歇性 `operation not permitted`；加打印就“好了” | 捕获前后 `dist.barrier()` + `acc.synchronize()`；tiny TP2 3/3 |
| replay padding 行写 slot -1 | 负索引写 KV 末尾，可能污染 | padding slot 填 0，context_lens 填 1 |
| 图 replay 在模型规模挂死 | 10 分钟 watchdog，`launch timed out and was terminated` | **未解**，见 §4.1 |
| 服务器 SSH 高 load 断连 | rsync `unexpected end of file`，ssh 255 | 重试；长任务 setsid nohup；短连接轮询 |
| 远端 `.git` 是伪仓库 | `git log` fatal | 不要在服务器上用 git；以本地为准，rsync 同步 |

---

## 7. 证据清单（服务器 `/root/`）

| 文件 | 内容 |
|---|---|
| `p6_step_overhead.json` / `.log` | 8B eager/graph 每步拆分 |
| `p6_moe_memory.json` / `.log` | T1：workspace 0 MiB + 每层 576 MiB |
| `p6_vendor_zero_copy.json` / `.log` | 零拷贝数值/显存/捕获 |
| `p6_moe_microbench2.json` / `.log` | 三后端 + mp22 + bmm_dense 表 |
| `p6_mccl.json` | 延迟 + 确定性 |
| `p6_mccl_capture_multi.json` | 96 次 all-reduce 图捕获 + eager 后 replay |
| `p6_mccl_eager.json` | 同上（eager after capture 版本） |
| `p5_graph_ab.log` | **30B 捕获成功** + 首次 replay 挂死 |
| `p5_graph_ab_8b.log` / `_8b2.log` | 8B A/B 挂死在 `verify graph_candidate` |
| `p6_graph_min_tp1.log` / `_tp2.log` | 最小捕获成功样本 |
| `p6_8b_3gen.log` | 8B 多次 generate 卡在 1/3 |
| `p6_replay_after_eager.log` / `_tp2.log` | tiny graph→eager→graph 不挂，但 token 变 |
| `p6_bmm_tiny.log` | tiny 六配置逐 token 全等 |
| `p2_grouped_mm.log` | loop/grouped/fused/TP1/TP2 全等 |
| `p6_rep_1..3.log` | tiny TP2 捕获 3/3 |

---

## 8. 接手自检（按顺序）

1. `mthreads-gmi | grep -E "^[01] "` → 两卡 0 MiB；若有残留，按 §1 清理。
2. 本地 `git status` → 确认 §3 的文件都在；**不要 reset**。
3. 读 §2.1 / §2.3 / §2.6 / §4.1，确认你知道“图 replay 未解、eager 路径才是主路径”。
4. 先做 §4.1 Step 0（8B 单次 generate ×5）建立复现率；同时可以并行准备 §4.2 的 30B eager 验证脚本（不占 GPU）。
5. 任何 30B 加载前：清理残留 + `rm -f /dev/shm/nanovllm*`；一次只跑一个。
6. 每个实验落盘 JSON + 日志 + sha256；不要只看 stdout。
7. 如果图 2–3 小时没修好，停止，转 §4.2–§4.4，把容量故事做完——这比一个挂死的图更有价值。

---

## 9. 一句话总结给用户

本轮把“图捕获为什么 OOM”从错误假设（厂商 workspace）纠正为真实原因（我们自己的 27 GiB 权重副本），并意外找到了一个更快且可捕获的 decode MoE（`bmm_dense`，1.3–1.95×），同时修掉了 shm 命令通道竞态。30B 图**捕获成功**，但**replay 在模型规模上间歇性挂死**，这是唯一未解问题，已给出 8B/TP2 的最小复现和 6 步二分计划。下一阶段的正确主线是：先在 30B 上落地 `bmm_dense + triton_paged + shm 修复` 并用 D2/在线容量验证，图只做时间盒诊断。
