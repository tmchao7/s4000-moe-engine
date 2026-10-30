# HANDOFF — 当前交接(滚动更新;本版 = P13 夜间迭代收官,2026-10-30)

> 本文是**唯一交接文档**(历史版本在 git 历史)。接手先读本文,再按 [AGENTS.md](../AGENTS.md) 核验环境与命令。规则不变:先写门再跑;一次一个 GPU 任务;负结果照录;本地 commit(不 push 除非用户授权);到点无论进度必须停下写交接、清服务器。
> 配套文档:[docs/PROMPT-P13.md](./PROMPT-P13.md)(任务书)、[docs/P13-survey.md](./P13-survey.md)(业界调研)、[docs/E-vendor-ticket.md](./E-vendor-ticket.md)(厂商工单,成文待发)、[docs/RESUME.md](./RESUME.md)(面试材料)。

## 0. 三十秒状态

- **仓库**:`feat/musa-port`;P13 夜间迭代全部本地 commit(未 push)。
- **服务器**:已清零核验(双卡 0 MiB、无进程、shm 干净、`/tmp/nanovllm-p13.lock` 已删)。`/root/moe-server-p6d`(pre-P8 旧树)**已删**(I30 定案后用毕)。证据 JSON 已拉回本地 `benchmarks/data/p13/`(sha256 见 §6);服务器 `/root/p13_data/` 保留全量日志与 meta。
- **生产栈(未变)**:`triton_paged` + `grouped_mm`(prefill)/`bmm_dense`(decode) + `enforce_eager=True` + `stall_free` + TP2/EP2 + chunk-256。**默认值全部未动**;`_SEG_BUCKETS`/`_SEG_PAD_RATIO` 两个新 env 开关默认惰性(见 §3-1)。
- **P13 一句话**:四个硬骨头全有终局——**I30 关闭**(ABA,方向反转)、**I33 关闭**(修复已验证但残余 AR 串行化超门 → 段图定格单流开关)、**I8/I26 闭合**(冷缓存塌=chunk 粒度)、**容量下界 1.070 req/s**;外加厂商工单成文。

## 1. P13 五项结论(报告 `benchmarks/P13-*.md`)

| 项 | 判定 | 关键数字 |
|---|---|---|
| **I30 定案**(ABA 三加载,23:34–00:53) | **关闭——方向反转** | 现树两档全部 12 repeat 配对快于 pre-P8 旧树 **23–51%**(1.0 档 0.2894 vs 0.5800/0.5877;现树 CV≤3%、双口径 100%;旧树 slo_standard 仅 ~75%);字面 INVALID 仅因旧臂 CV 8.5–12.1%;P9 的 0.387 与 v4 0.327 = 机时/单跑伪象 |
| **I33 padding 探针**(0.6B,22:45–23:17) | 判定力不足,机制转 fix A/B 判定 | b16→b17 阶跃仅 +3.3%(0.6B 设备计算太小);segment 臂 CV≤0.6%、跨模式 token 全等 |
| **I33 修复验证 + 分支关闭**(密桶+回退,08:45–11:45) | F1 PASS / **F2 FAIL(+4.7%>2%)** / F3 FAIL → **服务化分支关闭** | 30B 捕获 11 桶 **1056 图/rank 零失败 37.06 GiB**;0.6B bitwise 门 10/10、回退精确触发;**padding 税证实并移除**(0.9 档 +14.7%→+2.3%,0.5 档 +7.3%→+1.0%);残余 +1.0~4.7% = P11 桶边界固有开销 = 每层 2 次 eager AR 串行化(I29 范畴);四档 SLO 全 100% |
| **I8/I26 长上下文**(chunk-256/1024 两臂,01:29–03:09) | **闭合(机理+解法)** | 冷缓存塌=chunk 粒度:chunk-1024 砍冷 TTFT p50 −56~57%(3.30/4.14/5.07→1.449/1.784/2.201 s),0.5 档冷态 90.6% SLO 过 ≤2s 门;1536+ 请求**零 429 零崩**(I26 端到端稳定);暖缓存两臂 100% |
| **eager 容量**(00:59–01:29) | **下界 1.070 req/s** | 1.0/1.2/1.4/1.6 档 ×3 全部双口径 100%、并发至 57、TPOT p95 ≤0.46;拐点未探到;高于 I7 旧栈 0.94 与段图臂 1.036 |

## 2. 已判决不要再碰(累计,新增 P13 三项)

投机解码 NO-GO;I19 融合换步时否定;整图 TP2 否决(I29);**分段图在线转正否决并修复后关闭(P12+P13:残余 AR 串行化,单流开关为止)**;**I30 回归主张否定关闭**;**I8 长上下文机理闭合**;split=0/chunk-2048 短 prompt 默认化否决(P10);fused RMSNorm/容量收缩/placement 不采用;host-sync/mixed-split 已采纳;paged 已转正;I31 首 batch 丢弃协议。

## 3. 下轮 Top-N

| 优先 | 项 | 说明 |
|---|---|---|
| 1 | 发厂商工单 | `docs/E-vendor-ticket.md` 成文待发(developer.mthreads.com 或 vllm-musa GitHub issues)。四个问题:引擎级图 replay 楔死、~2048 图可执行上限求证、grouped_mm 捕获姿态、watchdog/图级诊断 |
| 2 | 饱和上界补测(可选) | eager 到 1.6×(1.070 req/s)仍 100%;`--fractions 1.8,2.0` 找拐点(~30 min,一发加载) |
| 3 | 1k 冷态 1.0 档(可选) | chunk-1024 冷态 1.0 档 TTFT p50 2.2 s 未过门;试 chunk-2048 冷态单点(~20 min) |
| 4 | 客户端改进 | `benchmark_online.py` 加 `--prompt-seed-per-repeat`(每 repeat 换 seed)——§4-1 暖缓存口径伪象的根治;改动小、零 GPU |
| 5 | (远期)I20 KV 量化 | 先做 torch_musa int8 转换带宽探针(30 min)再立项;参照 IPEX-LLM/OpenVINO 路线(见 P13-survey §3) |

**不要做**(已关闭或已否决):分段图服务化方向(AR 进图需整图,I29)、norm/rope 单点融合、EAGLE3(前置 I19)、chunk-256→2048 默认化(P10 门)。

## 4. 本轮新 SOP/坑

1. **跨 repeat 的 unique prompt 是确定性重建的**(`benchmark_online.py:180-195`,prompt 含 seed+request_id)→ rep2/3 全部命中前缀缓存。**"三 repeat 中位"在 unique 模式度量的是暖缓存路径**;冷容量看 rep1 或等 seed-per-repeat 功能落地。在线报告永远按 repeat 拆开看。
2. **tiny 模型不能跑 triton_paged**(head_dim=128/page=256/GQA 约束);tiny 上 sdpa 图有 I27 漂移(graph≠eager)→ **跨模式 bitwise 门只能在 0.6B paged 上下**。冒烟测试断言错在这里误报过一次。
3. **服务停掉后 /dev/shm/nanovllm_* 可能残留**(rank 子进程清理未走完)→ 停服务前扫 `/proc/<pid>/fd` 记 owned shm,kill 后删(`p13_run.py`/`p13_fix_chain.py` 已实现);残留时 `fuser` 确认无人持有再删。链里 idle() 拦住这种残留是**正确行为**,不要绕过。
4. **旧栈 repeat 噪声大**(0.87 qps 高载档 CV 8–12%):跨栈 A/B 必须同 session ABA 包夹;旧臂 CV 超限会把判定打成 INVALID——方向性结论看配对方向,正式门看 CV。
5. (继承)EADDRINUSE 端口竞态 flake 单独重跑;MaxStartups 冷却 60–90 s;锚定 pgrep;api_server 默认 ≠ 生产栈(必须 `--decode-moe-backend bmm_dense`);ssh 偶发 DNS 解析失败/网关瞬断 = 重试即可。

## 5. 红线

见 [AGENTS.md](../AGENTS.md);重申:默认值全部不动(`enforce_eager=True`、`NANOVLLM_SEGMENT_GRAPH=0`、`_SEG_BUCKETS=default`、`_SEG_PAD_RATIO=0`);段图分支已关闭,不再为其开实验;远端伪 `.git` 绝不提交;本地 commit 不 push 除非授权;一次一个 GPU 任务;离线图门(若将来任何按桶捕获复活)**必须含非桶边界 batch 点**。

## 6. 证据索引(本地 `benchmarks/data/p13/`)

| 文件 | sha256 前缀 | 内容 |
|---|---|---|
| `aba_A1/A2_B.json` ×3 | `d8b75297…`/`d5072a3c…`/`d30ae384…` | I30 三加载矩阵 |
| `aba_verdict.json` | `c476e929…` | ABA 判定(两档字面 INVALID,关闭条件满足) |
| `capacity.json` | `ddb96829…` | eager 容量 1.0–1.6× |
| `long256.json`/`long1024.json`/`long256_repeat.json` | `06fceda8…`/`4b8d663e…`/`eb00ec55…` | 长上下文两臂+暖缓存复测 |
| `padding_verdict.json` | `a41cf607…` | 0.6B padding 探针 |
| `fix_eager.json`/`fix_segment.json` | `3a254c72…`/`c37b62af…` | 修复 A/B 两臂 |
| `fix_verdict.json`/`fixprobe_verdict.json` | `c37ad217…`/`081cbd25…` | 修复判定/0.6B bitwise 门 |

冻结门与五字段报告:`benchmarks/P13-{i30,padding,fix,longctx,capacity}.md`;执行器:`scripts/p13_run.py`(aba/serving 相)、`scripts/p13_fix_chain.py`(修复相,`--resume-from` 支持续跑);服务器侧全量日志 `/root/p13_data/*.log`、meta json(含源码 sha256)。
