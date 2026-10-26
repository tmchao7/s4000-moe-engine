# HANDOFF — moe-server-s4000（MUSA S4000 双卡推理引擎）

> 最后更新：2026-10-26（会话 #3：P2 性能验收收尾 + P3/P4 设备侧压测 + 报告与提交）
> 项目：nano-vllm 移植到摩尔线程 MUSA S4000，目标 Qwen3-30B-A3B-FP8 TP2 推理。

---

## 0. 给新会话的最短开工指令

```bash
# 1) 连服务器（免密，中继偶发瞬断，长任务必须 nohup/setsid）
ssh -p 28036 root@connect.sha1.seetacloud.com
cd /root/moe-server-s4000        # 服务器上【没有 .git】，本地 Mac 仓库是唯一源头

# 2) 先确认环境干净（跑任何测试前都要做）
ps -eo pid,cmd | grep -E "python3 (tests|scripts)|spawn_main" | grep -v grep
rm -f /dev/shm/nanovllm
mthreads-gmi | grep -E "^[01] .*MiB"      # 两卡必须是 0MiB

# 3) 便宜的门先跑（tiny，1-2 分钟），全绿才碰 30B
PYTHONPATH=/root/moe-server-s4000 python3 tests/p2_grouped_mm.py   # 四路一致 + 重复确定性
PYTHONPATH=/root/moe-server-s4000 python3 tests/p3_scheduler.py    # FCFS vs stall_free
PYTHONPATH=/root/moe-server-s4000 python3 tests/p4_api.py          # API 契约（含 BatchEncoding）
PYTHONPATH=/root/moe-server-s4000 python3 tests/p4_async_engine.py # 异步并发/取消

# 4) 30B 验收 + 性能（一次加载 ~10 分钟，别重复加载）
PYTHONPATH=/root/moe-server-s4000 nohup python3 scripts/p2_benchmark.py \
  --batches 1,2,4,8,16,32,64 --prompt-len 128 --output-len 32 --repeats 3 \
  --moe-backends loop,grouped_mm --scheduler-policy fcfs \
  --verify-prompts --reference-outputs benchmarks/p2-reference-8tok.json \
  --verify-repeats 3 --verify-backends loop,grouped_mm \
  --output /root/p2_ablation_full.json > /root/p2_ablation_full.log 2>&1 &
```

- 服务器：AutoDL 容器，`/root/moe-server-s4000/`（代码），模型 `/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8`、`/root/models/Qwen3-8B`
- tiny 测试模型：`/tmp/tiny_moe`（随机权重 2 层 8 专家 float32，由 `tests/p2_tp_check.py` 自动重建；**没有 chat template**，不能用来验证 HTTP 服务）
- 本地仓库：`feat/musa-port`（`origin` = github.com/tmchao7/moe-server-s4000）

---

## 1. 项目硬成果（截至会话 #3）

| 阶段 | 结果 | 文件 |
|---|---|---|
| P0 基线 | vllm-musa 0.29.1 + Qwen3-8B TP2：8.4 tok/s（c=32，不随并发扩展） | `benchmarks/P0-vllm-musa-baseline.md` |
| P1 MUSA 移植 | Qwen3-8B 单卡 113.5 tok/s（batch 64），首 token 与 HF 100% 对齐 | `benchmarks/P1-musa-port.md` |
| P2 MoE 双卡 | 30B-A3B TP2+EP2 正确；**74.0 tok/s @b32、120.4 tok/s @b64**；grouped-MM 比 loop 快 **3.0–3.2×** | `benchmarks/P2-moe-tp2.md` |
| P3 调度 | FCFS vs stall-free（decode-first mixed prefill）+ 前缀缓存 / 专家放置 | `benchmarks/P3-scheduler.md` |
| P4 服务化 | OpenAI 兼容 API（SSE、取消、`/metrics`）+ 开放环压测 | `benchmarks/P4-serving.md` |

---

## 2. 会话 #3 完成的修复（都有设备侧证据）

### 2.1 grouped-MM 贪心不可复现 → 已修（本轮最重要）

- **现象**：同一进程内重复生成同样 4 条 prompt（temperature=0），`grouped_mm` 偶尔给出不同 token（近并列处）；`loop` 后端 3/3 稳定；`grouped_mm` 在 `set_moe_backend` 后的**第一次**前向也与稳态不同。
- **根因**：`_forward_grouped_mm` 用 `out.index_add_(0, flat_token, contribution)` 合并 top-8 贡献。同一 token 的 8 个 slot 命中同一行 → 重复索引 → MUSA 上走原子累加，**顺序由 kernel 调度决定**；bf16 下几 ulp 差异即可翻转近并列 argmax。`loop` 每个专家内 token 索引唯一，不触发原子累加，所以确定。
- **修复**：贡献先无累加 scatter 回 `(token, slot)` 布局，再沿 slot 维固定顺序求和：
  ```python
  dense = contribution.new_zeros(num_tokens, self.top_k, hidden)
  dense.view(num_tokens * self.top_k, -1)[flat_slot] = contribution
  return dense.sum(dim=1)
  ```
- **代价**：最坏（16k token prefill）多 ~512 MiB 临时缓冲；b32 吞吐 74.0 vs 修复前 75.4 tok/s（同量级）。
- **验证**：30B 上 `loop`/`grouped_mm` 各 3 次重复 + 预热**全部一致**，且 **loop↔grouped 4/4 条 prompt 逐 token 相同**。
- 副作用：旧参考 `p2_outputs.json`（原子累加时代的输出）作废，新参考写入 `benchmarks/p2-reference-8tok.json`；prompt 3 的贪心首 token 由 `\n\n\n```python` 变为 `当然可以！下面是一个带有详细注`（两者都连贯，新的才可复现）。

### 2.2 API 无法接真实 tokenizer → 已修

- transformers 5.x 的 `apply_chat_template(tokenize=True)` 返回 **`BatchEncoding`**（不是 list），直接喂给 `Sequence` 会坏。已实测确认（`TYPE BatchEncoding LEN 2`）。
- 修复：`api_server.py` 把返回值统一归一化成 `list[int]`（兼容 `BatchEncoding` / Tensor / 嵌套 list），空 prompt 直接 400。
- 测试：`tests/p4_api.py` 现在同时覆盖「返回 list」和「返回 BatchEncoding」两种 tokenizer；真实 30B 上 3 次同样请求输出完全一致。

### 2.3 其他收尾

- `scripts/p2_benchmark.py`：新增 `--verify-prompts`（先在进程内跑 4 条固定 prompt 的正确性/确定性门，再测性能）、`--cache-modes cold,hot`（一次加载测冷/热前缀缓存）、`--jsonl`；`ScheduledBatch` 兼容旧脚本。
- `scripts/benchmark_online.py`：每个负载点抓一次 `/metrics`，把 `engine_mixed_steps_total` 等计入 summary —— 调度收益有引擎侧证据，不只是端到端数字。
- `scripts/render_benchmark.py`：支持 `cache_mode` 维度、`--accept-batches`、`--markdown-out`。
- 删除死配置 `enable_profiling`（只透传、无人使用）。
- 新增 `tests/p4_benchmark_client.py`：用 stdlib 桩 SSE 服务在 **CPU** 上验证在线压测客户端（tiny 没有 chat template）。

---

## 3. 代码地图与关键实现

```
nanovllm/
├── config.py              # TP/EP、moe_backend、expert_placement、scheduler_policy、metrics_path
├── async_engine.py        # 后台线程 + 命令队列；服务化入口
├── engine/llm_engine.py   # step 循环、请求级/步级指标、--metrics-path JSONL
├── engine/scheduler.py    # FCFS / stall_free / stall_free_lpm；返回 ScheduledBatch
├── engine/block_manager.py# paged KV + block-hash 前缀缓存 + KV 计数器
├── engine/model_runner.py # musa 设备、shm 多进程、prefill/decode/mixed 批构造
├── entrypoints/openai/api_server.py  # /health /v1/models /v1/chat/completions /metrics
├── models/qwen3_moe.py    # packed experts + grouped-MM + 确定性 top-k 合并 + EP
└── layers/attention.py    # paged prefill/decode（decode 把 KV 尾部 mask 成 0，防 NaN 传播）
```

- **专家放置**：`linear`（0-63 / 64-127）、`round_robin`、`profiled`（`scripts/profile_experts.py` 生成静态均衡映射，带 `config.json` sha256 守卫，防止串用别的模型的地图）。
- **调度**：`stall_free` 每步先保留全部 decode，再用剩余 token budget 填 `prefill_chunk_size`（默认 256）的 prefill chunk；`stall_free_lpm` 在其上按最长前缀匹配挑选 waiting 请求，并带 starvation guard。
- **指标**：`LLMEngine.metrics()` / `Scheduler.metrics()` 暴露 step 数、prefill/decode token 数、各 mode step 数、抢占次数、KV 占用/前缀命中/分配回收；`--metrics-path` 按 step 输出 JSONL（`schedule_s` / `forward_s` 可分开看）。

---

## 4. 验收门（顺序执行，任一层失败都不要看性能数字）

| 层 | 测试 | 期望 |
|---|---|---|
| 逐层数值参考 | `tests/p2_ground_truth.py` | tiny TP1 与手写 CPU 参考逐 token 一致 |
| 并行等价 | `tests/p2_tp_check.py` | TP1 == TP2 |
| 后端/放置等价 | `tests/p2_grouped_mm.py` | TP1/TP2 × loop/grouped + round_robin 五路一致 + 重复确定性 |
| 解码确定性 | `tests/p2_determinism.py` | 反复重建引擎结果相同 |
| 调度等价 | `tests/p3_scheduler.py` | stall_free 出现 `mixed` step 且输出与 FCFS 相同 |
| 指标契约 | `tests/p3_metrics.py` | ScheduledBatch 形状 + KV 计数器 |
| API 契约 | `tests/p4_api.py` | health/models/stream/non-stream/metrics + BatchEncoding |
| 异步引擎 | `tests/p4_async_engine.py` | 并发完成 + 取消 + 计数器 + 干净退出 |
| 压测客户端 | `tests/p4_benchmark_client.py` | 桩 SSE 下分位数/成功率/`/metrics` 抓取正确 |
| 30B 端到端 | `scripts/p2_benchmark.py --verify-prompts` | 两后端确定性 + 跨后端逐 token 一致 + 输出连贯 |

---

## 5. 运维手册（每条都是血泪）

1. **进程清理**：`mp.spawn` 子进程不会自己退。每次跑完/失败后：
   ```bash
   pkill -9 -f "[s]pawn_main"; pkill -9 -f "tests/p[0-9]_"; pkill -9 -f "api_serve[r]"
   rm -f /dev/shm/nanovllm        # 残留会导致下次 barrier 卡死
   mthreads-gmi | grep -E "^[01] .*MiB"   # 必须 0MiB
   ```
2. **`pkill -f` 自匹配**：`ssh host 'pkill -f foo'` 会杀掉自己（远端 shell 命令行含 `foo`）→ 用括号技巧 `"fo[o]"`。
3. **SSH 中继不稳**：scp 大概闪断；**tar over ssh 流**实测稳定（`COPYFILE_DISABLE=1 tar -czf - . | ssh ... 'tar -xzf -'`）。后台长任务用 `setsid nohup ... &` 且 `ssh -f -n` 发起，**不要**让任务依赖会话；轮询用短 `ssh 'tail'`。
4. **AppleDouble 垃圾**：从 macOS 传包会带 `._*`，远端 `find . -name "._*" -delete`。
5. **MUSA 调试三铁律**：hook 里对比张量前必须 `torch.musa.synchronize()` 且先 `.contiguous()`；别信 strided view 的 bulk 拷贝；`get_rope` 有 `lru_cache`，换配置/换进程前 `get_rope.cache_clear()`。
6. **30B 加载 ~10 分钟**（FP8→bf16 反量化，CPU 侧热点），显存到位不代表加载完成，看日志里的 `[verify]`/`[p2-weights]` 才准。
7. **block 大小**：`kvcache_block_size` 必须是 256 的倍数；前缀缓存按 256-token 整块命中，**短 prompt（<256 token）不会命中**，测缓存收益要 `--prompt-len 512 --shared-prefix-len 384` 这种量级。
8. 项目目录里的 `core_*.mudmp` 是 MUSA core dump，可删。

---

## 6. 已知问题 / 下一步

- **注意力**：仍是 SDPA + 每步 `k_cache[blocks]` gather，未接 Triton-mtgpu paged attention kernel；这是 decode 的主要优化点。
- **CUDAGraph**：仍 `enforce_eager`，torch_musa graph capture 未验证。
- **小 batch 固定开销**：grouped-MM 的 `argsort`/`bincount`/两次 `_grouped_mm` 使 b1–8 加速比（2.7–2.8×）低于 b16–64（3.0–3.2×）。
- **温度**：本轮吞吐用 `temperature=0`；与外部数字对比时要统一口径。
- 30B 的**跨进程**逐 token 一致性未直接测量（同进程内两端确定性 + 跨后端 4/4 一致已测；修复原理上消除了原子累加，跨进程应当稳定）。
- 长期：Triton paged attention / fused MoE、W4/AWQ（GLM-4.5-Air 106B）、graph capture。

---

## 7. 产物清单

```
benchmarks/P2-moe-tp2.md          # 正确性门 + loop/grouped 消融（1-64）+ 显存 + 确定性修复
benchmarks/P3-scheduler.md        # FCFS vs stall-free 在线对比
benchmarks/P4-serving.md          # 真实 30B OpenAI 接口开放环压测
benchmarks/p2-reference-8tok.json # 30B 8-token 贪心参考（确定性构建）
benchmarks/data/                  # 原始 JSON/JSONL 与 PNG（.gitignore）
scripts/p2_benchmark.py           # 离线消融 + 正确性门 + 缓存冷热
scripts/benchmark_online.py       # 开放环压测 + /metrics 抓取
scripts/render_benchmark.py       # 表格 + PNG + 验收门
scripts/profile_experts.py        # 专家路由画像 → 静态均衡映射
tests/                            # 上述所有回归测试
docs/architecture.md              # 架构 + 可观测性 + 基准方法论
```

---

## 8. 给接手的 AI：待办清单（照做即可）

### 8.0 先读什么 / 别做什么

- **先读 §5 运维手册**，否则会重复踩坑（进程清理、ssh 中继、270s 加载、block=256）。
- **已经做完、不要重跑**（数字已冻结）：
  - P2 消融 batch 1–64 × loop/grouped × 3 重复 → `benchmarks/P2-moe-tp2.md`
  - 正确性门全套（§4）→ 全绿
  - 在线 FCFS / stall-free 各 4 个负载点 → `benchmarks/P3-scheduler.md`、`P4-serving.md`
- **不要改验收阈值**（`scripts/render_benchmark.py` 里 ≥1.5×、p95 TPOT 比 ≤1.10、CV ≤5%）。不达标就如实写 FAIL，不要调参让它 PASS。
- **任何 30B 性能测量之前必须先跑 §4 的门**。30B 一次加载 ~10 分钟，一次错误的测量等于浪费 10 分钟。
- 服务器上**没有 .git**；本地 Mac 仓库 `feat/musa-port` 是唯一源头，本次工作已提交（最近三个提交：MoE 模型与正确性 / 引擎与 API / 文档与结果）。

### 8.1 环境速查

```bash
ssh -p 28036 root@connect.sha1.seetacloud.com
cd /root/moe-server-s4000                       # 代码（无 .git）
# 模型
/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8   # 30G，主力
/models/Qwen3-8B                                          # 16G，dense 基线
/tmp/tiny_moe                                             # tiny，随机权重，无 chat template
```

**本地 → 远端同步**（scp 不可靠，用 tar 流）：

```bash
cd <local repo> && COPYFILE_DISABLE=1 tar -czf - \
  --exclude=.git --exclude=.zcode --exclude=__pycache__ --exclude='benchmarks/data' . \
| ssh -p 28036 root@connect.sha1.seetacloud.com \
  'cd /root/moe-server-s4000 && tar -xzf - && find . -name "._*" -delete && python3 -m compileall -q nanovllm tests scripts && echo COMPILE_OK'
```

**长任务启动模板**（`ssh -f -n` + `setsid`，中继断了也不影响）：

```bash
ssh -f -n -p 28036 root@connect.sha1.seetacloud.com \
  'cd /root/moe-server-s4000 && setsid nohup env PYTHONPATH=/root/moe-server-s4000 \
   python3 <script> <args> > /root/<name>.log 2>&1 < /dev/null &'
# 轮询（短连接，别 sleep 在 ssh 里）
ssh -p 28036 ... 'tail -5 /root/<name>.log; ps -o pid,etime -p <pid>'
```

**注意**：`scripts/benchmark_online.py` 的 `print` 没有 flush，重定向到文件时是块缓冲 —— 日志会在结束时才出现，期间看 `/metrics` 或 `ps` 判断进度。

### 8.2 任务 B1：前缀缓存冷/热对比（1 次加载 ≈ 20 min）

```bash
ssh -f -n -p 28036 root@connect.sha1.seetacloud.com \
 'cd /root/moe-server-s4000 && setsid nohup env PYTHONPATH=/root/moe-server-s4000 \
  python3 scripts/p2_benchmark.py --batches 8 --prompt-len 512 --output-len 32 --repeats 3 \
    --moe-backends grouped_mm --shared-prefix-len 384 --cache-modes cold,hot \
    --output /root/p3_prefix_cache.json > /root/p3_prefix_cache.log 2>&1 < /dev/null &'
```

- **必须** `prompt_len ≥ 512` 且 `shared_prefix_len ≥ 256`：前缀缓存按 256-token 整块哈希，短 prompt 命中率为 0（本轮在线压测里 `kv_prefix_hit_tokens_total` 一直是 0 就是这个原因）。
- 预期：`hot` 相对 `cold` 的 TTFT p95 明显下降（省掉 1 个 block 的 prefill）。
- 写回 `benchmarks/P3-prefix-placement.md` 的"前缀缓存"小节，并更新 `docs/roadmap.md`。

### 8.3 任务 B2：专家放置 linear vs profiled（2 次加载 ≈ 35 min）

1. **先在 tiny 上验证哈希守卫**（1 分钟，别跳过）：
   - `python3 scripts/profile_experts.py --model /tmp/tiny_moe --output /tmp/tiny_map.json`
   - 用该 map 起 LLM（`expert_placement="profiled"`）→ 应成功；
   - 故意改 `model_config_sha256` → 应 assert 失败。守卫在 `nanovllm/models/qwen3_moe.py::_expert_ids`。
2. **生成 30B 映射**（1 次加载）：
   ```bash
   python3 scripts/profile_experts.py --batch 32 --prompt-len 128 --output-len 64 \
     --output /root/p2_expert_map.json
   ```
   记录 `mean_balancedness`（越接近 1 越均衡）。
3. **测 profiled 放置**（1 次加载，参数必须与 P2 的 linear 完全一致才能比）：
   ```bash
   python3 scripts/p2_benchmark.py --batches 8,16,32 --prompt-len 128 --output-len 32 \
     --repeats 3 --moe-backends grouped_mm --expert-placement profiled \
     --expert-map-path /root/p2_expert_map.json --output /root/p3_profiled.json
   ```
4. 对比基准（linear，同参）：`benchmarks/P2-moe-tp2.md` 的 grouped 行 —— b8 24.21、b16 42.68、b32 74.01 tok/s。
5. 写回 `benchmarks/P3-prefix-placement.md`。**结论可能是"没有显著差异"**，如实写，并解释（EP2 下每卡 64 专家、hidden replicated，负载均衡对 all-reduce 前的工作量影响有限）。

### 8.4 任务 B3：把在线并发拉满 + 长 prompt（1–2 次加载 ≈ 30 min）

本轮在线最高只堆到 ~14–20 并发（达成 0.55 req/s ≈ 35 tok/s），低于离线 batch-32 的 74 tok/s。要直接对齐：

```bash
# 服务
python3 -m nanovllm.entrypoints.openai.api_server --model <30B> \
  --served-model-name qwen3-30b-a3b --scheduler-policy stall_free
# 拉满并发（目标 qps 1.6 → Little's law 下并发 ≈ 32）
python3 scripts/benchmark_online.py --model qwen3-30b-a3b --max-qps 1.6 \
  --requests 32 --max-tokens 64 --fractions 0.5,0.75,1.0 \
  --output benchmarks/data/online_highqps.json
```

第二个实验（更能体现 stall-free 收益）：把 prompt 拉长到 ~1k token。

```bash
python3 scripts/benchmark_online.py ... --prompt-words 1024 --max-tokens 64 \
  --fractions 0.5,0.75,1.0 --output benchmarks/data/online_longprompt.json
```

- 预期：prefill 成本从 ~0.2s 涨到 ~2s，FCFS 的 decode stall 变成主要延迟来源，stall-free 的 E2E/TPOT 收益应当从个位数百分比变得明显。
- 写回 `benchmarks/P3-scheduler.md` §3（把"短 prompt 收益小"的结论升级为两条 workload 的对比）。

### 8.5 任务 B4（可选）：30B 跨进程一致性（1 次加载）

本轮只测了**同进程内**确定性（loop/grouped 各 3 次 + 预热全一致、跨后端 4/4 一致）。若要把"跨进程也稳定"写实：跑两次 `p2_benchmark.py --verify-prompts ...`（可以只跑 `--batches 1`），比对两份 JSON 里 `verification.backends.*.runs[0]` 的 token ids。一致就补进 `benchmarks/P2-moe-tp2.md` §2，删掉 HANDOFF §6 里"跨进程未直接测量"的说明。

### 8.6 收尾约定

- 新数字**必须**用 `scripts/render_benchmark.py` 生成表格（别手抄），原始 JSON/JSONL 放 `benchmarks/data/`（已 gitignore）。
- 同步更新：`docs/roadmap.md` 勾选、`README.md` 的进度表、`benchmarks/README.md` 的文件索引。
- 提交分三类（可参考本轮）：① 模型/正确性 ② 引擎/服务/工具 ③ 文档/结果。**不 push**，除非用户明确要求。
- 提交前至少跑：`python3 -m compileall -q nanovllm tests scripts`；有 GPU 就再跑 §4 的 tiny 门（`p2_grouped_mm` / `p3_scheduler` / `p4_api` / `p4_async_engine` / `p4_benchmark_client`）。
- 收工前清理服务器：`pkill -TERM -f "api_serve[r]"`、`pkill -9 -f "[s]pawn_main"`、`rm -f /dev/shm/nanovllm`、确认 `mthreads-gmi` 两卡 0MiB。
