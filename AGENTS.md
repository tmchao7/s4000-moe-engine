# AGENTS.md — 本仓库的 Agent 工作入口

基于 nano-vllm 的摩尔线程 S4000(双卡)MoE 推理引擎:Qwen3-30B-A3B-FP8,TP2/EP2,MUSA 全栈(torch_musa + MCCL + Triton-mtgpu)。生产栈:`attention=triton_paged`、`moe_backend=grouped_mm`(prefill)、`decode_moe_backend=bmm_dense`、`enforce_eager=True`、`scheduler=stall_free`、chunk-256。30B eager b1 decode 步时 142 ms。

## 文档地图(只有这 5 个是活文档)

| 文档 | 职责 |
|---|---|
| [docs/HANDOFF.md](docs/HANDOFF.md) | **接手先读这个**:当前状态、已判决清单、未完成、下轮 Top-N、SOP |
| [docs/PLAN.md](docs/PLAN.md) | 本次迭代要干什么 + 全部冻结门(均不可改) |
| [docs/ISSUES.md](docs/ISSUES.md) | 问题台账:I 系列(技术)/S(投机解码,已关)/E(生态厂商) |
| [docs/architecture.md](docs/architecture.md) | 架构 |
| [docs/numerics-policy.md](docs/numerics-policy.md) + [docs/musa-pitfalls.md](docs/musa-pitfalls.md) | 数值契约(五层门)/ MUSA 平台坑 |

历史轮次报告在 `benchmarks/archive/`、一次性探针在 `tests/archive/`、`scripts/archive/`(git 历史可查;旧交接文档见 `git log --oneline -- docs/`)。

## 常用命令

```bash
# 服务器(ssh s4000b):代码在 /root/moe-server-s4000,伪 .git,绝不在远端 git 操作
# 本地→远端同步(tar over ssh;scp 偶发不稳):
tar czf - nanovllm tests scripts | ssh s4000b 'tar xzf - -C /root/moe-server-s4000'

# 开跑前核验(一次只允许一个 GPU 任务;发现陌生进程不杀、只记录):
ssh s4000b 'pgrep -fa "^/usr/bin/python3|^python3|^[s]pawn_main" || echo idle; mthreads-gmi | grep MiB'

# 四套常跑单测(都在服务器跑;本地 Mac 无 pytest):
ssh s4000b 'cd /root/moe-server-s4000 && python3 -m pytest tests/test_benchmarking_v3.py -q'
ssh s4000b 'cd /root/moe-server-s4000 && PYTHONPATH=. python3 tests/p8_scheduler_gaps.py'
ssh s4000b 'cd /root/moe-server-s4000 && PYTHONPATH=. python3 tests/p2_grouped_mm.py'      # 六配置全等 rc=0
ssh s4000b 'cd /root/moe-server-s4000 && PYTHONPATH=. python3 tests/p6_bmm_dense_tiny.py'  # 六配置全等 rc=0

# 启动 30B 服务(热 ~7 min、冷 11-12 min;判活用 /health + NANOVLLM_TRACE_INIT 的 [init] 行):
NANOVLLM_TRACE_INIT=1 PYTHONPATH=. python3 nanovllm/entrypoints/openai/api_server.py \
  --model /root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8 [--prefill-chunk-size 2048]

# 在线压测(见 --help;双 SLO、--repeats、--mode concurrency、--prompt-tokens N --prompt-mode unique):
PYTHONPATH=. python3 scripts/benchmark_online.py --model <path> --tokenizer <path> \
  --mode poisson --max-qps 0.87 --fractions 0.5,0.75,0.9,1.0 --requests 128 --prompt-words 128 --repeats 3

# 崩溃/楔死后清理(楔死形态:双卡自旋或单 rank 死+另一 rank 旋;按 PID kill,不 pkill -f):
ssh s4000b 'rm -f /dev/shm/nanovllm*; mthreads-gmi | grep MiB'   # 须双卡 0 MiB 再继续
```

## 红线(违反会毁掉一轮)

- 本地 commit,**不 push**;不 reset;远端 `.git` 是伪仓库,绝不在远端 git 操作。
- 不改 `site-packages`;不改 `scripts/p5_decide.py` 阈值(冻结门)。
- 不用 `vllm_fused`/`mp22` 默认;不全局 `moe_backend=bmm_dense`(prefill OOM)。
- `pgrep/pkill -f` 会匹配到执行命令的 shell 自身 → 用 `[p]` 括号技巧或按 PID;pgrep 必须锚定 `^/usr/bin/python3|^python3|^[s]pawn_main`。
- 一次一个 GPU 任务;长任务 `setsid nohup` + lock + 全量日志落盘(观测管道会吞崩溃现场);不留自动重启 waiter。
- 代码先行:改引擎的实验,代码+单测+tiny 验证必须在加载 30B **之前**完成;30B 加载窗口只做测量。
- 不重跑已定案实验(ISSUES/HANDOFF 的已判决清单);文档与实测冲突时以代码和实测为准。
- `enforce_eager=True` 是生产默认;30B 图 A/B 被 I29 阻断,禁止启动(复活条件见 ISSUES I29)。

## 平台 SOP(血泪,详见 docs/musa-pitfalls.md 与 docs/HANDOFF.md §4)

- 运行时切 eager/graph/backend/attention 一律 `runner.call("set_xxx", ...)`(双 rank 同步);写本地属性 → MCCL 死锁楔死双卡。
- MUSA 快照三件套:D2H 前 `synchronize()`;跨块步长视图拷贝前 `.contiguous()`(I28);先 device 侧 `.item()` 对照再下"内存写坏"结论。
- 失败的 engine init 会泄漏 process group:重试前 `dist.destroy_process_group()` + `gc.collect()` + `empty_cache()`。
- A/B 噪声协议:交错 ≥5 次、中位数、CV≤5%,否则判 INVALID;跨加载 A/B 用同一冻结负载 + 臂内 CV 门。
- 单问题 45 min:写"已排除/未排除/下一步",切下一条;失败/负结果原样记录,不伪造成功。
- 到约定时长即使没做完也必须停下、写交接、清服务器(双卡 0 MiB + shm 干净 + 无进程)。
