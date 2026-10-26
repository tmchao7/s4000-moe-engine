# HANDOFF-P6 — 图捕获交接（2026-10-26 20:25 收手）

> 写给接手的 agent。**先读 §0 和 §1，再决定做哪条**。
> 本文件只记录**已经跑出来的事实**；标【已实测】的可以直接引用，标【待验证】的必须自己复现后再下结论。
> 旧文档 `docs/P6-graph-capture.md` 已按本轮实测重写；上游调研见 `docs/P6-musa-graph-survey.md`。

---

## 0. 一句话现状

**decode 图捕获在算子层面已被证明可行**（Triton paged attention、厂商 Triton fused MoE（含 EP2 expert_map、1/2/8/32/64 token 各桶）、MCCL all-reduce 全部实测可捕获且 replay 数值完全一致）；
**但把整模型包进图时连续 3 次在"捕获前的 warmup forward"处 OOM**，最后定位到 **厂商 `vllm_fused` MoE 在 decode 时一次性要掉约 15.6 GiB 显存**（权重只占 28.6 GiB / 卡，另有 16.35 GiB 空闲，仍在"分配 384 MiB"时失败，说明这次 forward 内部累计申请了 ~15.6 GiB）。
**因此下一步不是继续调参，而是二选一：修掉厂商 MoE 的这 15.6 GiB，或换一个不带大 workspace 的可捕获 MoE（`bmm` 稠密路径已实测可捕获）。**

---

## 1. 我停在哪一步 / 服务器上还留着什么

| 项 | 状态 |
|---|---|
| 最后一次运行 | `scripts/p5_graph_ab.py`（20:15 启动），在 `capture_cudagraph` 的 **bucket=1 warmup forward** 抛 `OutOfMemoryError: Tried to allocate 384.00 MiB`，脚本已按设计把失败写进 `/root/p5_graph_ab.json`（字段 `capture_error`）后退出码 2 |
| GPU | `mthreads-gmi` 两张卡 **0 MiB / 0%**，无残留进程，`/dev/shm/nanovllm*` 已清 |
| 代码 | 本地已提交到 `git log`（见 §3）；服务器 `/root/moe-server-s4000` 与本地同步（rsync，`benchmarks/data` 除外） |
| 可直接复用 | `/root/d2_run.sh`（前缀缓存探针 + b32/b64 稳定性，两次加载，20:2x 已确认 `bash -n` 通过）；`/root/p6_case_*.json`（逐算子探针证据）；`/root/p6_verbose_*.log`（失败 case 的完整 traceback） |

**没有做完的事**：30B 上的 graph A/B（1 次都没成功）、前缀缓存 cold/hot（第 4 发加载遗留）、b32 不可复现性定位。

---

## 2. 本轮已完成、已验证的事实（可直接用，不要重跑）

### 2.1 在线容量 v2（第 5 发加载，**已完成**）

证据：`benchmarks/data/online-v2.json`（1.2 MB，512 条请求）、`.jsonl`、`.md`；报告：`benchmarks/P5-serving-v2.md`。
栈：sdpa + grouped_mm + `enforce_eager=True` + `stall_free`；输入 137 token，输出 64 token，Poisson 开放环，SLO = TTFT≤2 s 且 TPOT≤500 ms。

| 负载档 | 输出 tok/s | TPOT p95 | SLO 达标率 | goodput |
|---:|---:|---:|---:|---:|
| 50% (0.435 qps) | 26.31 | 0.455 s | **100%** | 0.411 req/s |
| 75% | 39.38 | 0.697 s | 26.6% | 0.163 |
| 90% | 42.13 | 0.685 s | 11.7% | 0.077 |
| 100% (0.87 qps) | 45.72 | 0.694 s | 7.0% | 0.050 |

**结论**：SLO 拐点在 50%→75% 之间，卡在 TPOT 而不是 TTFT；并发 19→45 只换来 1.74× 吞吐。→ 抬容量只能降单步 decode 成本。

### 2.2 图捕获可行性（逐算子，**已完成**，含对旧结论的推翻）

证据：`benchmarks/data/p6_graph_probe.json`；每个 case **独立进程**跑（关键，见 §7.2）。

| case | 结果 | 备注 |
|---|---|---|
| `plain` / `triton_copy` / `triton_paged` / `sdpa_decode` / `bmm_batched` | ✅ max_rel 0.0 | **`triton_paged` 可捕获**（旧文档说不行，是错的） |
| `vendor_fused_moe` | ✅ | 厂商 Triton MoE，E=64/H=2048/N=768，32 token，mp22 关闭 |
| `vendor_fused_moe_ep` | ✅ | 带 **EP2 `expert_map`**，**1/2/8/32/64 token 每个桶各一张图**全部数值一致 |
| `mccl_all_reduce` | ✅ | 2 rank，每层一次 EP all-reduce 可入图 |
| `aten_grouped_mm` / `_relaxed` | ❌ | `torch._grouped_mm` → `operation not permitted when stream is capturing`；`capture_error_mode='relaxed'` 无效 |
| `route_pack` | ❌ | **`torch.bincount` 不可捕获**（其余 cumsum/argsort/index_select/index_add_ 未单独验证） |
| `host_sync` | ❌（预期） | `bool(tensor.any())` |

**顺带修正的三个旧错误**（写在 `docs/P6-graph-capture.md` §2）：
1. 旧探针把 4 个 case 放在**同一个 try** 里，`grouped_mm` 抛异常后 `paged` 根本没执行 → "Triton 不可捕获"是假结论。
2. 旧探针在捕获区内执行 `int(context_lens.max())`（主机同步）→ 自己把 paged 弄失败。
3. `torch.bincount` 需要主机侧尺寸，必然不可捕获。

### 2.3 已合入的代码改动（默认行为不变）

| 提交 | 内容 |
|---|---|
| `dcea378` | 探针重写（逐 case 隔离 + `root_error` 错误链）、`is_capturing()`、`store_kvcache` 捕获安全、MoE 早退捕获安全、`decode_moe_backend`、`scripts/p5_graph_ab.py`、`tests/p5_d2_stability.py` |
| `e798f46` | `benchmarks/P5-serving-v2.md`（在线容量 v2） |
| `14d9954` | P3/P4 指标措辞、`benchmarks/P5-profiling.md`、P6 上游调研 |
| 之前 | `de8f135`（矩阵抢救 + JSONL 流式）、`8023881`（attention A/B 裁决 keep_sdpa）、`ef60c3e`、`7e0c451`、`293b02e` |

**默认值未变**：`attention_backend=sdpa`、`moe_backend=grouped_mm`、`expert_placement=linear`、`enforce_eager=True`。`decode_moe_backend` 默认 `None`。

---

## 3. 剩下的任务（按优先级，带命令与验收）

### T1（第一步，30 分钟，无加载）搞清楚那 15.6 GiB 是什么

已知数据（来自 `/root/p5_graph_ab.log` 与失败 traceback）：

```
[kvcache] rank0 blocks=32 kv=0.38 GiB total=47.90 used=31.55 peak=31.01 current=28.57 free=16.35 GiB
Traceback ... model_runner.py:383 outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # bucket=1 warmup
[graph-ab] CAPTURE FAILED: Tried to allocate 384.00 MiB ... 44.13 GiB allocated, 282 MiB free
```

即：**1 token 的 warmup forward 吃掉了 ~15.6 GiB**，随后连 384 MiB 都拿不到。
`MUSAGraph()` 本身不占显存（已实测：`/root/pool_probe.py`，空卡上 `MUSAGraph()` 前后 free 不变，tiny capture 只耗 0.1 GiB），**所以不是图池，是 MoE**。

下一步排查（照这个顺序）：

```bash
# 1) 看厂商 MoE 的 workspace 分配：新版 vLLM MoE 有 workspace13 / workspace2，按 max_num_tokens 定尺寸
grep -rn "workspace13\|workspace2\|max_num_tokens" /usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe.py | head -30
# 2) 我们的调用是 stateless 的 fused_experts（nanovllm/models/qwen3_moe.py::_forward_vllm_fused），
#    看它是否用模块级默认 max_num_tokens（可能是 16384）来建 workspace，且按 layer 缓存 -> 48 层累加
# 3) 一分钟实验（不加载 30B）：在空卡上连续跑 48 次 fused_experts(1 token)，每次后打印 memory_allocated，
#    看是否 +0.3 GiB/次 -> 就能确认是"每层一个 workspace 且被持有"
```

验收：能指出"哪一行代码、按什么尺寸、被谁持有"；或能给出把 workspace 尺寸压到 decode 桶（≤64 token）的改法。

### T2（与 T1 并行可做）用 `bmm` 稠密专家实现可捕获 MoE（备选，已实测可捕获）

- 事实：`torch.bmm` 捕获+replay **数值完全一致**（`bmm_batched` case，E=64/T=32/H=768/N=2048）。
- 设计（写进 `nanovllm/models/qwen3_moe.py`，新增 `moe_backend="bmm_dense"`，**只给图内 decode 用**）：
  1. `local_ids = global_to_local[topk_ids]`，`valid = local_ids >= 0`（**不要** `bool(valid.any())`，捕获时跳过该分支）；
  2. 取本步**被激活的本 rank 专家** `active = unique(local_ids[valid])`（用 `torch.unique`，可捕获；若不可捕获则按 E 固定 gather）；
  3. `w = gate_up_proj[active]`（E_act×H×2N）、`x` gather 到每专家固定 `T_pad`（`T_pad` 取捕获桶上限，如 `max_bs*top_k/E_act` 向上取整到 8）；
  4. `torch.bmm(x_pad, w)` → SwiGLU → `torch.bmm(act, w2)` → masked 加权 scatter（用 `dense.view(bs*top_k,-1)[flat_slot]=...; dense.sum(1)` 的确定性写法，已在 `_forward_grouped_mm` 里）；
  5. 不允许出现 `.item()/tolist()/bool()/bincount`。
- 验收：新后端与 `grouped_mm` 在 decode 短输出（≤32 token）上 token 一致率 100%（用 `benchmarks/p2-reference-8tok.json` 做门），且在 `tests/p6_graph_probe.py` 里能捕获（写一个 `case_bmm_dense_moe` 先证明，再进引擎）。

### T3（T1 或 T2 完成后）跑 30B graph A/B

```bash
# 服务器；默认 sdpa+grouped_mm 的对照组会跑，候选组用 vllm_fused（或 T2 的 bmm_dense）
cd /root/moe-server-s4000 && export PYTHONPATH=/root/moe-server-s4000
> /root/p5_graph_ab.log
PYTHONPATH=/root/moe-server-s4000 PYTHONUNBUFFERED=1 setsid nohup \
  python3 scripts/p5_graph_ab.py --batches 1,2,4,8,16,32 --repeats 3 \
  --gpu-memory-utilization 0.72 --num-kvcache-blocks 32 \
  --decode-moe-backend vllm_fused --prefill-moe-backend grouped_mm \
  --attention-backend triton_paged --output /root/p5_graph_ab.json \
  > /root/p5_graph_ab.log 2>&1 < /dev/null &
# 判决（冻结门，禁止改阈值）
python3 scripts/p5_decide.py --mode graph --input /root/p5_graph_ab.json \
  --output benchmarks/p5-graph-ab-decision.json --markdown /root/p5_graph_ab.md
```

- 门（`scripts/p5_decide.py::GRAPH_GATES`）：token 全等、b1 ≥ +15%、b8 ≥ +5%、b32/b64 ≥ 0.98、CV ≤5%、显存增量 ≤2 GiB。
- **必须分开读的两组比值**：`graph_candidate / eager_candidate`（同内核，只差有没有图 → 这才是图收益）；`eager_candidate / eager_default`（换内核的代价：`sdpa→triton_paged` 是 +19~59%，`grouped_mm→vllm_fused` 是 −3~25× 的坑，见 §4）。
- 若 b32/b64 OOM：把 `--max-num-seqs 8`（桶变成 1/2/4/8）再跑一次，**只捕小桶**也是合法结论（大 batch 本来不是 launch 瓶颈）。
- 产出：`benchmarks/P5-graph-ab.md` + `benchmarks/p5-graph-ab-decision.json`，并更新 `docs/P6-graph-capture.md` §6。

### T4（独立，40 分钟，脚本已就绪）补第 4 发加载的两个欠账

```bash
bash /root/d2_run.sh    # 内部：p2_benchmark 前缀探针(1 次加载) + tests/p5_d2_stability.py(1 次加载)
```

- 前缀缓存 cold/hot：`--probe-batches 8 --probe-prompt-len 512 --probe-shared-prefix-len 384`，数字写进 `benchmarks/P5-final-matrix.md` §3（共享前缀必须 ≥256 token 才命中整块）。
- b32 不可复现：`tests/p5_d2_stability.py --repeats 10 --bisect` 会跑 2×2 开关矩阵（moe × attention）；b64 崩溃单独复现并记录 `dmesg`/traceback。
- 验收：能回答"b32 不稳定到底来自 grouped_mm / SDPA / MCCL / KV 尾块"中的哪一个；答不出也要写清已排除项。

### T5（收尾）文档

1. `docs/HANDOFF.md`：加载台账补第 5–7 发、更新"已完成的实测结论"、把 §9 的下一步换成 T1–T4。
2. `benchmarks/README.md`：加 `P5-serving-v2.md`、`P5-graph-ab.md`。
3. `docs/P6-graph-capture.md` §6/§7 填入 A/B 真实数字（现在只有方法，没有结论）。

---

## 4. 关键数字（不要重新测，直接引用）

**离线矩阵**（sdpa+grouped_mm，prompt 128 / out 32，schema v2，`benchmarks/P5-final-matrix.md`）：
b1 3.06 · b2 5.70 · b4 11.44 · b8 22.57 · b16 40.57 · b32 69.39 · b64 111.75 tok/s；CV ≤0.024；b64 第 3 次重复崩溃（MUSA unknown error）。

**attention A/B**（`benchmarks/P5-attention-ab.md`，裁决 `keep_sdpa`）：
tok/s sdpa→triton：b16c512 35.68→42.44（1.19×）、b32c512 53.98→71.66（1.33×）、b16c2048 20.41→29.29（1.44×）、b32c2048 24.82→39.48（1.59×）；长上下文 TPOT 比 0.652/0.597；**b32 两个后端都不可复现**（首次分歧在第 2 个 token）；跨后端 token 一致率 0.969（b32c512）/0.740（b32c2048）。

**厂商 MoE（vllm_fused）单次调用耗时**（默认配置，未 autotune）：1 token 1.36 ms、8 7.06、16 10.07、32 13.51、64 20.40、256 82.19、1024 262.64 ms；mp22 fast path 在 1024 token illegal memory access（默认关闭）；手写融合 `fused_grouped_mm` 与 grouped 在 bf16 上 22/24 一致 → 都不能当默认。

**profiling**（`benchmarks/P5-profiling.md`）：长上下文 attention 占 forward 31.5–58.6%，MoE 27.5–53.5%，EP all-reduce ≤5.45%（门槛 20%，拒绝 overlap），rank 不平衡 ≤5.19%（门槛 10%，拒绝冗余专家）。

---

## 5. 环境与连接

```bash
ssh -p 28036 root@connect.sha1.seetacloud.com          # 已配免密
# 仓库（服务器）：/root/moe-server-s4000       模型：/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8
# 本地仓库：/Users/tmchao7/repos/02_personal/moe-server-s4000
rsync -az --exclude '.git' --exclude 'benchmarks/data' --exclude '__pycache__' --exclude '*.mudmp' \
  ./ root@connect.sha1.seetacloud.com:/root/moe-server-s4000/ -e 'ssh -p 28036'
```

环境：2× MTT S4000 48 GB（PCIe，无 NVLink）、驱动 5.2.0、MUSA SDK 5.2.0、torch 2.11.0.post1 + torch_musa、triton 3.2.0（backend `mtgpu`）、transformers 5.13、vllm-musa 0.29.1.dev0+g98dff2a.musa、py3.10。
一次 30B 加载 ≈ **10 分钟**（宿主机 page cache 命中时更快）；`gpu_memory_utilization=0.85` 时权重后用掉 ~31.5 GiB、KV 按公式填满，**图捕获需要额外预留**（见 §7.4）。

---

## 6. 本轮新发现但**尚未解决**的问题

1. **`vllm_fused` 在 decode 时一次性申请约 15.6 GiB**（§3 T1）。这是当前图捕获的唯一硬阻塞。怀疑是 vLLM 新版 MoE 的 `workspace13/workspace2` 按 `max_num_tokens`（可能是 `max_num_batched_tokens=16384`）分配，且 48 层各持一份。
2. **`torch.bincount` 不可捕获**：`_forward_grouped_mm` 用它算 per-expert offsets；若将来要让 grouped_mm 进图，必须改成 `zeros(E).index_add_(ids, ones)`。
3. **b32 decode 不可复现**（两个 attention 后端都中招），仍是 attention 收益落地的唯一前置。
4. **b64 第 3 次重复崩溃**（MUSA unknown error → rank1 退出 → DistBackendError）：矩阵数据靠 JSONL 抢救；`p2_benchmark` 现已每点落盘。

---

## 7. 坑（每条都真实踩过，别再踩）

1. **`pkill -f` 自杀**：远端命令里只要出现 `p5_graph_ab` / `p6_graph_probe` 字面量，`pkill -f "[p]5_graph_ab"` 会匹配到**执行这条命令的 bash 自己**（因为整条 command line 含该串），于是把自己杀掉、日志为空。→ 要么别 pkill，要么用 `pgrep -f "^python3 scripts/p5_graph_ab"` 这种**锚定**模式，或先 `pgrep -fa` 看 PID 再 `kill -9 <pid>`。
2. **探针必须每个 case 一个进程**：一次失败的 capture 会把流/上下文搞坏，后续 case 只会收到掩盖型报错 `operation failed due to a previous error during capture`。
3. **捕获区内禁止主机同步**：`.item()`、`.tolist()`、`bool(t.any())`、`if tensor:`。`int(context_lens.max())` 这类计算要挪到 `with graph` 外面。
4. **KV 显存与图捕获抢内存**：`allocate_kv_cache` 按 `gpu_memory_utilization` 把显存填满 → 捕获必 OOM。已加 `num_kvcache_blocks` 显式覆盖（`p5_graph_ab.py --num-kvcache-blocks`），A/B 用 32（=0.38 GiB）。
5. **`MUSAGraph()` 不占显存**（`/root/pool_probe.py` 实测），别再怀疑图池；tiny capture 只耗 0.1 GiB。
6. **本机没有 `timeout` 命令**（macOS）；远端长任务用 `setsid nohup ... &` + 短连接轮询。
7. **`scripts` 包被 site-packages 遮蔽**：脚本里用 `sys.path.insert(0, parent)` + `from p2_benchmark import ...` 这种按文件路径导入。
8. **bash heredoc 没走 ssh stdin** 会写出 0 字节文件 → 用 `scp` 传脚本，或 `ssh 'cat > f' < local`。
9. **清理仪式**：`pkill -9 -f "[s]pawn_main"; pkill -9 -f "api_serve[r]"; rm -f /dev/shm/nanovllm*`；spawn 出来的 rank 进程命令行不含脚本名，需要 `pgrep -fa "spawn_main"` 或看 `mthreads-gmi` 的 PID 再杀。
10. **一次只跑一个 GPU 任务**：探针会污染 benchmark 数字（本轮差点在在线 benchmark 期间跑探针）。

---

## 8. 不要做的事

1. 不要改 `scripts/p5_decide.py` 里的任何阈值；不要"看完结果再改门"。
2. 不要在门没过的情况下改默认值（`enforce_eager` 保持 `True`、`attention_backend=sdpa`、`moe_backend=grouped_mm`）。
3. 不要重跑：profiling（第 1 发）、attention A/B（第 2 发）、上游 MoE 评估（已离线做完）、在线容量 v2（第 5 发已完成）。
4. 不要把 `vllm_fused` 设成 prefill 的默认后端（prefill 会慢 3–25×，TTFT 直接崩）。
5. 不要 `git push`（用户明确要求）。
6. 不要在上一发加载没结束时就起下一发（显存/端口冲突）。

---

## 9. 接手自检清单（逐条打勾再动手）

- [ ] `ssh -p 28036 root@connect.sha1.seetacloud.com 'date; mthreads-gmi | grep -E "^[01] "'` → 两卡 0 MiB、无残留进程；
- [ ] `cd /root/moe-server-s4000 && git log --oneline -3` 与本地一致（本地 HEAD 应含本文件所在提交）；
- [ ] 读 §2.2 的探针表，确认"可捕获/不可捕获"清单没有更新；
- [ ] 选一条：T1（查 15.6 GiB）或 T2（写 `bmm_dense`）；
- [ ] 任何 30B 加载前跑一次 §7.9 的清理；
- [ ] 每次只改一个变量，结果写进对应 `benchmarks/*.md`，并附证据 JSON 路径 + sha256。
