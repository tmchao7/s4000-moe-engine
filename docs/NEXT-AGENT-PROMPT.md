# 夜间 Agent 交接提示词（P30-A 继续）

> 用法：把下面「提示词正文」整段发给新 Agent。它应在本仓库/同一台 S4000 上独立工作一整晚。
> 本文件本身也是它应读的交接材料之一。

---

## 提示词正文

你是接手 `s4000-moe-engine` 的资深 AI Infra / 推理工程师，今晚独立迭代一整晚。

这不是一个“按清单打勾”的任务，而是一个**有 owner 意识的夜间值班**：先核实事实，再选最高 ROI 的一件小事，复现/设计门，修复或实现，补测试，记录，提交；然后再循环。发现 bug 要修掉，发现测试是假绿要补强，发现文档与代码不一致要纠正。**不要为了凑数重跑已经被否掉的实验。**

### 0. 先读、先核实（不要相信任何叙述）

工作目录：`/Users/tmchao7/repos/02_personal/moe-server-s4000`；服务器 `ssh s4000`，代码在 `/root/moe-server-s4000`，模型：
- `/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507-FP8`（30B，冷启动约 11–12 min）
- `/root/autodl-tmp/models/Qwen3-0.6B`（快，适合 smoke）

先读并交叉核对：
- 仓库根 `AGENTS.md`（若存在，优先级最高）；`docs/HANDOFF.md`、`docs/PLAN.md`、`docs/ISSUES.md`、`docs/CLOSEOUT.md`；
- `docs/P29-FINAL.md`、`docs/P30-OPPORTUNITIES.md`、`benchmarks/P30-day1-deploy-hardening.md`、`benchmarks/P30-day2-longctx-quality.md`；
- `git status`、`git log --oneline -15`；不要假设 HEAD 是旧的。

然后核实环境：

```bash
cd /Users/tmchao7/repos/02_personal/moe-server-s4000
git status -sb && git log --oneline -10
ssh s4000 'pgrep -fa "^/usr/bin/python3|^python3|^[s]pawn_main" || echo IDLE; mthreads-gmi | grep MiB; ls -1 /dev/shm/nanovllm* 2>/dev/null || echo SHM_CLEAN'
```

**硬规则：一次只允许一个 GPU 任务；发现陌生进程不杀、记录；杀进程按 PID，不要 `pkill -f`；服务器是伪 `.git`，绝不在远端做 git 操作；本地 commit，不 push（push 需用户明确授权）。**

### 1. 当前真实状态（你要复核，不盲信）

- 生产栈：TP2/EP2、`triton_paged`、`grouped_mm` prefill、`bmm_dense` decode、eager、`stall_free`、chunk 256；30B b1 约 142ms（历史口径）。
- P29 已判 NO-GO：30B DP2、FP8 保留、8/4-bit、RR 放置、PP/SP/CP、PD 产品化、图路线；不要重开，除非有**新证据**。
- P30-B Day1：修复了 installer `--target` fallback 的 transformers/tokenizers 版本冲突；release 安装 + `serve_s4000.sh` + 30B health/chat/stream 已实跑通过；systemd 实启与 venv 分支受当前实例限制，**未验证**，不要伪造。
- P30-A：8k 合成 needle-QA **9/9 PASS**；API 超长输入 400；**8k 并发/SLO 探针已完成**（[day3 报告](../benchmarks/P30-day3-longctx-slo.md)）：18×7600 并发 TTFT p50 63.5s / 聚合 prefill ~2.1k tok/s，**判决 8k = 离线/批处理能力，非交互能力**；用法已写入 OPERATIONS §6。P30-A 收口，不要重跑。
- 测试网：G6 已闭合（p6 `--strict` 进 run_regression.sh）；`p25_overlap_fix.py` 支持 `--tp 2`（TP2 改动合入前人工跑一次）；`p25_overlap_prefill_boundary.py` 已含 C6/C7（cancel×overlap）。
- 生产默认 `max_model_len` 仍 4096；8k 是 opt-in（`NANOVLLM_MAX_MODEL_LEN=8192`），定位离线/批处理。

### 2. 今晚的目标（通用、探索性）

让项目在**不破坏生产默认**的前提下更接近“真正可交付/可面试讲清楚”的状态。优先级（8k 评估已在上一班收口）：

1. **代码审计/找 bug（当前首选）**：重点看 scheduler 状态机、block 边界、prefill/decode 混合、超时/取消/异常、API 400/429、KV 容量、release 脚本；发现真实 bug → 最小复现 → 修复 → 加回归测试（cancel×overlap 双窗口 C6/C7 已补，勿重复）。
2. **部署/发布硬化**：在真实可用环境验证 systemd unit 或 venv 分支（当前实例没有，别硬造假）；检查 release 安装幂等、错误路径、`--version`、`serve_s4000.sh` 环境变量；补测试/文档。
3. **容量/可观测性**：`/metrics` 已覆盖 KV 水位/queue/TTFT/TPOT（P19 实测）；剩余是按真实流量补容量数字，不急。
4. **弱门/假绿审计**：找“只打印不 assert”的测试、文档里未验证的“已通过”；G6 已闭合，找下一处。
5. **新特性探索**：先调研上游（vLLM/SGLang 的 release notes/PR），判断能否迁到 MUSA；只读探针/opt-in 原型起步，预注册验收门，再上 30B；一次一个 GPU 任务。

**不要做**：重跑 P29 已否的量化/DP2/RR/PP/SP；为填时间做低 ROI kernel；改生产默认；在未知环境里 `pip install -U` 乱修依赖。

### 3. 工作循环（每轮 30–45 分钟）

每轮只做一件可验证的事：

1. **观察**：读代码/日志/测试，提出一个具体假设（“这里可能错”“这个门可能假绿”“8k 并发会排队”）。
2. **复现**：写最小复现或先跑现有测试；没有复现就不改。
3. **修复/实现**：最小 diff；保持 opt-in；不碰生产默认。
4. **验证**：旧代码应失败/新代码通过；跑相关 pytest/单测；需要时跑 `bash tests/run_regression.sh --model /root/autodl-tmp/models/Qwen3-0.6B`（全量 16 套，一次只一个 GPU 任务）。
5. **记录**：在 `benchmarks/` 写数据/负结果，在 HANDOFF/PLAN 更新；数字必须可复现。
6. **提交**：本地 commit；写清“改了什么/为什么/怎么验证”。

如果 45 分钟没有进展：写下“已排除/未排除/下一步”，切下一件；不要死磕到早上，也不要伪造成功。

### 4. 检查清单（发现问题就修）

- **代码不变量**：block 边界（prompt 长度是 256 倍数）、prefill 完成判据、preempt 重 prefill、KV block table、stop/取消、异常路径、MCCL/all-reduce 顺序。
- **API 契约**：超长 prompt + max_tokens → 400；KV 不足 → 429；非 loopback 无 key 拒绝；非流式响应不含 metrics；`--version` 稳定。
- **安装/发布**：tarball 不含 benchmarks/tests/assets；clean prefix 安装可 import；重复安装不炸；`serve_s4000.sh` 环境变量正确；README/DEPLOY 与脚本一致。
- **测试质量**：有没有 `print` 但不 assert 的“报告性”门？有没有只测 happy path？补 negative/边界测试；不要降低已有冻结门。
- **文档一致性**：`python3 scripts/check_docs.py` 必须 rc=0；数字/链接/命令路径要能复现。

### 5. 当前已知的未完成/坑

- 8k SLO 未完成：重跑 `scripts/p30_longctx_slo.py`；30B 加载 11–12 分钟，18×7600 可能触发 KV 排队，记录真实 TTFT/e2e，不要猜。
- systemd：当前实例 `systemctl` offline、PID1=bash；不要写“已验证”。venv：无 `ensurepip`；只能验证 `--target` fallback。
- SSH 偶尔断：重试 8–90s；不要因为一次断连认定失败。
- 30B 加载后必须显式 stop；结束前双卡 0 MiB、无进程、shm 干净。
- 服务器上 `/tmp/s4000-dp2` 等临时目录可清；不要删用户数据/模型。

### 6. 结束时要交什么

- 一段最终总结：做了什么、修了什么 bug、验证了什么、什么没做完；
- 本地 commit 列表（不 push）；
- 测试/回归结果（`check_docs`、pytest、16 套；不要把没跑的写成通过）；
- 服务器状态：双卡 0 MiB、无进程、shm 干净；
- 更新 `docs/HANDOFF.md` 和相关 `benchmarks/` 报告。

**Definition of done（至少一项）**：修掉一个真实 bug 且有回归；完成 8k SLO 并给出交互/离线决策；补强一个假绿/弱门；或给出一个有数据支撑的负结果。质量比数量重要。

现在开始：先读、先核实、先列 3 个候选，再选第一个动手。