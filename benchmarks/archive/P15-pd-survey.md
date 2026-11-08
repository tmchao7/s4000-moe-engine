# PD 分离(两台 S4000)技术调研与实施方案

- 日期:2026-10-29
- 状态:**调研完成,待评审**。本文档是独立调研件,不属于 AGENTS.md 文档地图的 5 个活文档;若立项,再按仓库流程转为轮次任务书。
- 关联仓库:
  - 参考实现 `<local-repo>/nanoPD`(下称 nanoPD,从零实现的 PD 分离引擎,CUDA/单机多卡)
  - 本引擎 `<local-repo>/moe-server-s4000`(下称 musa-nano-vllm,MUSA/双 S4000 TP2)
- 证据分级:【官方文档】【代码确认】【论文】【媒体/论坛】【推算】。全部外链见 §11。

---

## 0. TL;DR

1. **可行性:成立,且生态证据比预期好。** 三个决定性事实(均为一手证据,§4):
   - Mooncake TransferEngine **已官方支持 MUSA**,PyPI 有预编译包 `mooncake-transfer-engine-musa`,vLLM-MUSA PR 已接入;
   - `MooreThreads/vllm-musa`(v0.28.0-dev)仓库**自带 MooncakeConnector 1P1D 分离示例**(单机双卡,RDMA+TCP 双模式),含 MUSA 专属补丁(KV-first FA、PD 双侧 KV 池分离);
   - 摩尔线程 × 硅基流动已发布 **PD 分离异构混部生产白皮书**(4×S5000 Prefill + 2×H200 Decode,Mooncake RDMA 跨机零拷贝传 KV,"传输时延对 TTFT/TPOT 无可见影响",已投产)。证明的是 **"MUSA 平台 + Mooncake RDMA 跨机传 KV"这条链路成立**,但卡型是 S5000,不是 S4000。
2. **难度:中等偏高(3.5/5)。** 控制面(调度/路由/状态机)有 nanoPD 现成蓝图,难度低-中;KV 导出/导入是接口工程,难度中;**不确定性集中在传输层与网络,而不是引擎逻辑**。MVP(0.6B 端到端)约 1.5–2 周;30B 生产级(含 A/B 判定)约 4–6 周。
3. **推荐路线 C(§7):** 在 musa-nano-vllm 上自研 PD 控制面(抄 nanoPD 作业),传输通道做成**可插拔**:首发 `TCP + pinned memory`(零新依赖,保底必通),M0 探针通过后接入 Mooncake TE MUSA wheel(先 TCP 模式,发现 peermem/RDMA 再升级)。**不做跨机 TP**:每台机内部维持现有 TP2+MCCL 不动,跨机只传 KV 和元数据。
4. **三个必须先实测的未知数(§4.4、M0):** ① S4000 驱动栈是否提供 `mthreads-peermem`(RDMA 直传显存的前提,白皮书只证明 S5000);② **两台机器之间的实际网络带宽**(若为云租户实例互通,可能只有 1–10 Gbps,直接决定 TCP 路线上限);③ `mooncake-transfer-engine-musa` wheel 对 S4000 的兼容性。
5. **收益边界要诚实(§1.2):** vLLM 官方文档明示 PD 分离 **"does NOT improve throughput"**,收益是 TTFT/尾延迟 SLO;我们已有 stall_free + chunk-256,干扰已被部分缓解。**动手前先用现有 metrics 量测 collocated 模式下 prefill 打断 decode 的 ITL 尾部**(Sarathi-Serve 结论:吞吐优先场景 chunked prefill 常胜,严格 SLO/长上下文场景 PD 胜出)。P13 长上下文实验(1K 输入、TTFT p50≤2s 门)正是收益敏感场景。

---

## 1. 背景与问题定义

### 1.1 PD 分离是什么

LLM 推理两个阶段资源画像相反:prefill 是计算密集(大矩阵乘,吃算力),decode 是访存密集(每步读全部 KV,吃显存带宽)。合部部署时两者互相干扰:一个长 prompt 的 prefill chunk 会把同批 decode 步的 ITL 打出尖刺。PD 分离把两阶段放到不同 GPU/节点上,prefill 算完把 **KV cache 搬给 decode 节点**,decode 只做自回归。

- DistServe(OSDI'24)提出 goodput(同时满足 TTFT 与 TPOT SLO 的有效吞吐),同硬件下满足 SLO 的请求数 **4.48×** 或收紧 SLO **10.2×**【论文】。
- Mooncake(OSDI'24,Kimi 生产系统):长上下文场景请求量 +75%,KV 传输用基于速率控制的拥塞控制【论文】。

### 1.2 收益边界与反动机(先算账再动手)

- vLLM 官方文档明确:PD 分离**不提升总吞吐**,收益是 TTFT/ITL 独立调优与尾延迟控制【官方文档】。
- 同届 OSDI 的 Sarathi-Serve 证明:**chunked prefill 能在共享池场景消除大部分干扰**;吞吐优先时 chunked prefill 常胜,严格 SLO/长上下文时 PD 胜出【论文】。
- 我们的生产栈**已经是 stall_free + chunk-256**(decode-first、bounded prefill chunk、KV 准入跳过防 HOL,`nanovllm/engine/scheduler.py:152-233`),即 Sarathi 路线已落地一半。PD 的边际收益来自:① chunk 粒度(256)与 decode 步(142ms)交替造成的残余 ITL 尖刺;② 1K+ 长上下文场景一个请求占 4–16 个 chunk budget 时的排队;③ P/D 资源画像匹配后各自吞吐上限提高。
- **结论:Step 0 应先用现有 step 级 JSONL + Sequence.metrics()(TTFT/TPOT/ITL,`nanovllm/engine/sequence.py:85-109`)量测当前负载的 ITL 尾部与 TTFT 分布,确认干扰真实存在且值钱。** 若 ITL p95 与 p50 差距 <2×,PD 大概率不划算,记录后收手。

### 1.3 为什么必须是"两台机器"

Qwen3-30B-A3B checkpoint 是 FP8,但 S4000 无原生 FP8 算力,加载期一次性反量化为 **bf16**(`nanovllm/utils/loader.py:28-35,104-112`)。bf16 权重 ≈ 61 GB,单张 S4000(48 GB)**装不下**,TP2 强制——所以"单机内卡 0 prefill / 卡 1 decode"对 30B 不可行,PD 只能跨机:每台机一个完整 TP2 引擎,机器间传 KV。0.6B 开发模型则可以单卡部署,是天然的 bring-up 载体。

---

## 2. 现状盘点

### 2.1 musa-nano-vllm 引擎现状(与本任务相关的事实)

| 项 | 现状 | 出处 |
|---|---|---|
| 进程模型 | uvicorn → AsyncLLMEngine 后台线程 → rank0(进程内)+ rank1(mp spawn 子进程);命令通道 = POSIX shm(1MiB)+ pickle + Event,`dist.barrier()` 收尾 | `nanovllm/engine/llm_engine.py:32-65`、`model_runner.py:137-202` |
| MCCL | `init_process_group("mccl")`,rendezvous 硬编码 `tcp://127.0.0.1:port` | `model_runner.py:88-93`、`llm_engine.py:19-29` |
| Scheduler | stall_free(生产默认):decode-first、bounded prefill chunk=256、KV 准入跳过;mode= mixed/decode/prefill | `scheduler.py:152-233` |
| KV cache | 单张量 `[2(K/V), 48层, num_blocks, 256, 2 kv头/rank, 128]` bf16 连续池;块大小 256;30B TP2 实测 723 块/rank ≈ 8.7 GiB/rank;**每块每 rank 12 MiB 连续** | `model_runner.py:503-514` |
| 每 token KV | 96 KiB/token(全模型)/ 48 KiB/token/rank | 推算,§5.1 |
| KV 导出/导入 | **完全没有**(全仓 grep export/migrat/disagg/recv/isend 零命中) | — |
| decode 接入面 | `prepare_decode` 只依赖 `block_table[-1]/last_block_num_tokens/last_token`,**天然可承接外来预填充好的序列**——PD 集成最有利的一点 | `model_runner.py:572-587` |
| MoE 后端 | 每步按 `context.is_prefill` 自动分流:prefill→grouped_mm,decode→bmm_dense;**PD 拆分后 P/D 节点天然各用各的**,且 bmm_dense 必须 `max_num_seqs≤64`(I32) | `models/qwen3_moe.py:512` |
| 运行时切换 | `runner.call("set_xxx")` 双 rank 同步(shm 广播 + barrier),可切 backend/attention/eager | `model_runner.py:185-231` |
| 图 | **I29 阻断,30B 图 A/B 禁止启动;P12 分段图在线判定 FAIL。D 节点只能 eager**(b1 decode 142ms) | AGENTS.md 红线 |
| 压测 | `scripts/benchmark_online.py` 打 OpenAI SSE,有 `--mode concurrency`、双 SLO、Wilson CI——**PD 架构前置一个同 API 路由层即可直接压测** | `benchmark_online.py:270-284,434-471` |
| 同机假设 | TP2 三处硬绑定同机:init 127.0.0.1、shm+Event 命令通道、rank1 spawn。**路线 C 不跨机 TP,三处全部保留不动** | §7.1 设计决策 DD1 |

### 2.2 nanoPD 参考实现剖析(控制面蓝图)

nanoPD 是单进程多线程的从零实现:CentralScheduler 统一派发,CollocatedWorker(GPU0)/ PrefillWorker / DecodeWorker 并发推进,自研 CUDA paged attention,块粒度 KV 传输 + 成本模型路由。

**分离路径请求生命周期**(关键机制,`nanoPD/router/central_scheduler.py:202-352`):

```
add_request → _dispatch_waiting(路由:collocated or disagg)
  disagg: 预检 decode 侧空闲块数 → 派给空闲 PrefillWorker 线程
→ prefill_batch_and_extract: 整段 prefill 一次前向 → 采样 first token
   → extract_kv_to_pinned: 按块 D2H 进 pinned buffer(同步)
→ first_token/block_table/pinned_buf 写入 _prefill_done
→ _flush_prefill_done: 所有权移交 DecodeWorker.receive_kv_async
→ DecodeWorker: 请求进 _pending(传输流异步 H2D 装载, event 跟踪)
   → 每个 step 开头 _promote_ready(): event.query() 完成的晋升到 running
   → 纯 decode 连续批,EOS 出批补位
→ 结果回写 _states → router.update(闭合在线学习环)
```

五个值得照抄的机制:

1. **块粒度传输 + 编号对齐**:`PinnedKVBuffer` K/V 形状 `(num_layers, num_block, num_kv_heads, block_size, head_dim)`(`kv_transfer.py:28-30`),buf 第 i 个 slot 对应 `block_table[i]`;单次 strided copy 搬"单块×全部层"。
2. **双流重叠**:`compute_stream` 跑前向、`transfer_stream` 收 KV,`event.query()` 非阻塞晋升,decode 永不等传输(`decode_worker.py:35-68`)。
3. **所有权移交接口极薄**:跨 worker 只传 `(block_table, num_computed_tokens, output_token_ids)` 三元组。
4. **成本模型路由**:`T_colloc = αL + β·max(1,load/thresh)·O + γ·L·min(load,thresh)/thresh` vs `T_disagg = αL + L·bytes_per_token/bw + …`,四参数(α/β/batch_thresh/γ/bandwidth)启动时在真机 profile 拟合(`cost_model/profiler.py:216-252`)。闭式判据:Disagg 胜 ⇔ γ/transfer_rate > batch_thresh/system_load。
5. **Oustput 长度预测器** + 路由在线回填(实现是滑窗均值,不是 README 说的 Bayesian)。

**已知短板**(移植时心里有数):分离路径无连续批、无 chunked prefill,`prefill_batch_and_extract` 末尾硬同步是串行瓶颈(文档自认的头号限制);P2P 可用时仍无条件 D2H;CoW 只重映射不拷数据;无传输超时/重试/校验和。

### 2.3 nanoPD 可复用性结论

| 模块 | LOC | 处置 |
|---|---|---|
| block_manager / sequence | 232 | **原样复用思想**(我们用自己的 BlockManager,但"块表即契约"的接口设计照抄) |
| Scheduler / Engine | ~245 | 控制流复用到 D 节点注入逻辑 |
| Router + Predictor + CostModel | 582+602 | **逻辑 100% 复用**,参数在 S4000 重测 |
| CentralScheduler | 446 | 状态机复用,"调函数"换成"发消息" |
| kv_transfer | 82 | 骨架保留(extract→relay→load),中间通道换网络(§7.5) |
| paged_attention csrc | 434 | **不需要**——我们已有 triton_paged,跳过整个 kernel 移植 |
| benchmark(Poisson 到达) | 1430 | 思路复用;实际直接用我们的 benchmark_online.py |

**nanoPD 对本项目最大的价值不是代码而是分层**:它把"传输"隔离在 82 行的 kv_transfer.py 里,两端只认块表和 event——这个切面正好是跨节点改造的替换点。

---

## 3. 业界方案对照

| 方案 | 传输层 | 调度/路由 | 与本项目距离 |
|---|---|---|---|
| **vLLM**(V1) | `--kv-transfer-config` 9 个 connector:NixlConnector(UCX/GDS)、LMCacheConnectorV1、MooncakeConnector(rdma/tcp 双模)、OffloadingConnector(CPU)、MultiConnector 等;`KVConnectorBase_V1` 挂 scheduler + model runner 双侧【官方文档】 | 1P1D + `mooncake_connector_proxy.py` 前置代理;支持 `--tokens-only` 无分词器 prefill 实例 | 接口设计(`KVConnectorBase_V1`)是 D 节点注入 API 的最佳参照 |
| **SGLang** | `--disaggregation-transfer-backend mooncake/nixl/ascend`【官方文档】 | **bootstrap/handshake 协议最完整**:decode 向 prefill 的 bootstrap server 心跳注册(5s 间隔)→ 槽位预留(early / prefill_complete 两种策略)→ 300s 三级超时;官方 router 带 `--pd-disaggregation` | 控制协议(§7.4)直接对标;异构 TP 需 staging buffer(吞吐 2–5×) |
| **Mooncake TE** | RDMA(主)+ TCP(基线),多 NIC 聚合、拓扑感知选路、故障换路;4×200G RoCE 实测 87 GB/s;**`-DUSE_MUSA=ON` + mthreads-peermem 官方支持,MUSA wheel 已上 PyPI**【代码确认】 | TransferEngine 只管传输,元数据服务 ETCD/Redis/HTTP 三选 | **路线 C 的传输层首选** |
| **NVIDIA Dynamo + NIXL** | NIXL:UCX 默认(自动落 RDMA/TCP)、GDS、GPU-initiated networking;KV-aware router 从 P/D 双池选路【官方文档】 | — | **backend 矩阵无 MUSA 证据,不应假设可行** |
| **production-stack** | router(L7)+ LMCache(经 NIXL) | `--prefill URL --decode URL` 双池;AWS SageMaker HyperPod 生产背书 | router 参考实现 |

**对选型的含义**:跨机 KV 传输在 x86 世界已经是"搭积木"问题,真正的分叉点只有一个——**MUSA 上传输层能走多快**(§4)。

---

## 4. MUSA 平台证据(决定传输层选型)

### 4.1 官方一手证据

1. **PD 分离异构混部白皮书**(摩尔线程 × 硅基流动,2026-08)【官方一手材料,最重要】:
   - 拓扑 4×S5000 Prefill + 2×H200 Decode,Kimi K2.6,P/D 两侧 FP8 格式完全一致;
   - KV 走 **Mooncake RDMA 直送对端显存,零拷贝**,layer-wise 计算与通信重叠(prefill 过程中提前传部分 KV);
   - 结论:"Mooncake RDMA 传输 KV Cache 的时延开销未对 TTFT/TPOT 产生可见影响",**已投入生产环境**。
   - 注意:证明的是 S5000,且引擎是硅基流动自研(非 vLLM/SGLang)。
2. **vllm-musa 内置 PD 示例**【代码确认】:仓库 `example/disaggregated_serving/`——单机双 MUSA 卡 1P1D,prefill 卡 `--kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'`,decode 卡 `kv_consumer`;另有 `mooncake_store_pd_serving.sh`(KV 落 Store,支持 SSD offload)。RDMA 容器前提:host 网络 + `/dev/infiniband` + memlock + `MC_FORCE_HCA=1`(RDMA 失败**报错而非静默回退 TCP**)。MUSA 适配层在 `vllm_musa/distributed/mooncake_compat.py`。README 未承诺跨机示例。
3. **Mooncake MUSA 支持**【代码确认】:issue #1883 记录 `[PG] Add MUSA build support`(#2329)、`[EP] support MUSA platform`(#2480,2026-05 合入)、`[MUSA] Add release-musa workflow`(#2576)均已合入;PyPI 包 `mooncake-transfer-engine-musa` 已发布;构建要求 "Install MUSA SDK and **mthreads-peermem** for GPUDirect RDMA"。
4. **MCCL 多机声明**【官方页面】:MUSA SDK 产品页原文"多 GPU 和多节点通信基元""双机通信 RDMA 带宽利用率高达 97%""双机 All Reduce 带宽可达 194GB/s";torch_musa README 标注 "MCCL(S4000 only)",仓库内有 `ProcessGroupMCCL` 与 14 个单测。但"用 MCCL 传 KV 做 PD"无公开案例,194GB/s 对应卡型/网卡未注明。
5. **SGLang-MUSA / llm-d**【官方文档】:SGLang 上游已含 MUSA 支持;llm-d 官方 P/D 指南兼容矩阵有 Moore Threads 行(vLLM 走 Mooncake **TCP**、SGLang 走 Mooncake RDMA,均 community 级)。

### 4.2 S4000 规格【媒体聚合,以实机为准】

| 项 | 数值 | 备注 |
|---|---|---|
| 显存 | 48 GB,768 GB/s | 我们实测 723 块×12MiB ≈ 8.7GiB/rank KV 池与之相容 |
| 算力 | FP16/BF16 ~100 TFLOPS(tensor),INT8 200 TOPS | |
| 互联 | MTLink 1.0 240 GB/s(卡间);PCIe 4.0 x16(多数来源,Gen5 两说) | **PCIe 代际需上机 `lspci` 确认**,决定 pinned 路径上限 |
| 发布 | 2023-12,第三代 MUSA;无原生 FP8(第四代 S5000 才有) | 与我们 loader 反量化 bf16 的事实互证 |

### 4.3 传输层选型证据表

| 通道 | 在 MUSA 上的证据 | 强度 / 不确定度 |
|---|---|---|
| Mooncake TE **RDMA** 跨机直传显存 | 白皮书(S5000 生产级);vllm-musa RoCE 容器要求;PyPI MUSA wheel;依赖 `mthreads-peermem` | **方案级强**;S4000 卡/驱动是否带 peermem = **高不确定** |
| Mooncake TE **TCP** 模式 | TE 原生支持;llm-d mthreads overlay 明确 vLLM-on-MUSA 走 "Mooncake TCP" | **中强**;S4000 上无公开吞吐数字 |
| MCCL 跨机 send/recv | 多节点基元 + 双机 RDMA 97%/194GB/s 官方声明;ProcessGroupMCCL 代码在 | **中**;send/recv 原语是否暴露、KV 场景无案例 |
| **TCP + pinned memory**(自研) | torch_musa 有 `PinnedMemoryAllocator`(代码确认);路径通用成熟 | **中**;上限受 PCIe 代际 + 网卡双重限制 |
| NVMe/文件中转(Mooncake Store) | vllm-musa `mooncake_store_serving.sh` 实测脚本(SSD offload) | 单机确认,跨机共享盘未证实 |
| NIXL(UCX/GDS) | **无 MUSA backend 证据** | 不应假设可行 |

### 4.4 未找到公开证据、必须上机实测的问题清单(M0 输入)

1. S4000 驱动栈是否提供 `mthreads-peermem`(或等价 GPU 内存 RDMA 注册能力);
2. 两台 S4000 之间网络的真实带宽/时延(HCA 型号?RoCE?还是仅普通以太网?**若是云租户实例互通,带宽可能只有 1–10 Gbps**);
3. Mooncake TE 在 S4000 上 `get_local_topology()` 的输出;MUSA wheel 安装与运行;
4. MCCL 双机 init 在 S4000 上是否成功(备用通道评估);
5. S4000 PCIe 实际代际(4.0 vs 5.0);
6. vllm-musa 的 MooncakeConnector 1P1D 跨机可用性(示例仅单机);
7. KV dtype 两侧逐位一致性(bf16,理论上同代码同 dtype,但传输校验和仍要抽查);
8. D 节点 48GB 内 30B TP2 + bmm_dense + max_num_seqs≤64 的显存余量(现网已在跑,风险低)。

---

## 5. 关键量化推导

### 5.1 KV 体积表(Qwen3-30B-A3B,bf16,TP2)

公式:`2(K+V) × 48层 × 4 kv头 × 128 head_dim × 2B = 96 KiB/token`(全模型;每 rank 持 2 个 kv 头 = 48 KiB/token)。

| 请求 prompt | KV 总量(双 rank) | 每 rank | 块数(256/块) |
|---|---|---|---|
| 256 | 24 MiB | 12 MiB | 1 |
| 1K | 96 MiB | 48 MiB | 4 |
| 4K | 384 MiB | 192 MiB | 16 |
| 8K | 768 MiB | 384 MiB | 32 |
| 32K(max_model_len 上限内) | 3 GiB | 1.5 GiB | 128 |

若未来 KV 量化到 FP8(I20,HANDOFF 在册课题)传输减半——**PD 初版不与 I20 耦合**。

### 5.2 传输时间与网络需求【推算】

单请求传输时间(prompt 长度 L):

| L | 有效载荷 | 25GbE TCP(≈2.8GB/s) | 10GbE(≈1.1GB/s) | 100G RDMA(≈12GB/s) |
|---|---|---|---|---|
| 1K | 96 MiB | 34 ms | 87 ms | 8 ms |
| 4K | 384 MiB | 137 ms | 349 ms | 32 ms |
| 8K | 768 MiB | 274 ms | 698 ms | 64 ms |

对照:decode 一步 142 ms(eager b1)——**1K 请求的 TCP 传输可与首批 decode 重叠,几乎免费;4K+ 请求 TCP 路径开始可见,RDMA 仍可忽略**(与白皮书结论同型)。

持续带宽需求(喂饱 D 节点):设 D 节点输出 X tok/s、平均输出长度 O、平均 prompt L:

```
需求带宽 ≈ (X / O) × L × 96 KiB
例:X=300 tok/s, O=128, L=1K → 226 MiB/s ≈ 1.9 Gbps(10GbE 无压力)
    X=300 tok/s, O=128, L=4K → 900 MiB/s ≈ 7.6 Gbps(10GbE 顶格,25GbE 稳)
```

**这组数字直接把 M0 的网络实测变成 go/no-go 门**:≥10 Gbps → TCP 路线可支撑 MVP;≥25 Gbps → TCP 路线可支撑生产;<10 Gbps → 必须 RDMA 或把 PD 限定在长 prompt 请求(按长度路由,§7.7)。

### 5.3 节点容量不对称

- **D 节点(eager,生产栈)**:b1 decode 142 ms;batch 增大步时变长,`max_num_seqs≤64`(bmm_dense 约束,I32)。粗上限 ~300 tok/s 输出。
- **P 节点(grouped_mm prefill,chunk-256)**:算力富余,prefill 吞吐显著高于 D 节点消纳速度(白皮书里 S5000 prefill 单卡也只有 H200 的 46–54%,但我们的瓶颈在 D 侧 eager decode)。
- 固定 1P1D(两台机)下 P 节点会周期性空闲——这正是 nanoPD 成本模型"自适应路由"的用武之地(后期优化,§7.7):P 节点空闲时把短请求留在本地 collocated 处理,等价于 1.5P1D。

---

## 6. 候选路线

### 路线 A:整体切换 vllm-musa + MooncakeConnector(生态路线)

vllm-musa 自带 KVConnectorBase_V1 + MooncakeConnector + 1P1D 示例,理论上改两个启动参数就能跑。
- 优点:PD 机制最成熟(vLLM V1),MUSA 官方适配,示例可抄。
- **代价:P0–P13 全部优化弃用**——grouped_mm/bmm_dense 双后端、stall_free、chunk-256、容量收缩、admission 指标都是我们自己的补丁;vllm-musa 对 Qwen3-30B-A3B MoE 的性能画像完全未知(30B b1 142ms 这条基线要重新挣);依赖重量级跳变。
- 定位:**只做 M0 的廉价验证(在测试机上跑通它的单机 1P1D 示例,作为"生态能跑"的证据与对照),不作为主线。**

### 路线 B:纯自研(musa-nano-vllm + 自建 TCP/pinned 通道)

不引入任何新依赖,通道自写(pinned relay + socket)。
- 优点:零依赖风险;nanoPD 的 82 行骨架就是模板。
- 缺点:重造 Mooncake 已解决的问题(多 rail、拓扑发现、拥塞控制、断线重连);RDMA 永远摸不到(peermem 注册、内存池这些活 Mooncake 已做完)。

### 路线 C(推荐):musa-nano-vllm 自研 PD 控制面 + 可插拔传输通道

控制面抄 nanoPD(状态机/块表契约/双流异步/成本模型),传输层定义统一接口,后端三选:`tcp_pinned`(首发保底)→ `mooncake_tcp` → `mooncake_rdma`(视 M0)。路线 B 是路线 C 在"M0 全部失败"时的退化形态,所以选 C 不需要赌 M0 结果。

| 维度 | A(vllm-musa) | B(纯自研) | **C(混合,推荐)** |
|---|---|---|---|
| 保留 P0–P13 优化 | ✗ | ✓ | ✓ |
| PD 功能工作量 | 最小(理论) | 中 | 中 |
| 传输层上限 | RDMA(若 peermem) | TCP 上限 | TCP 保底,RDMA 可插 |
| 新依赖 | 整个 vLLM | 无 | mooncake wheel(可选,接口隔离) |
| 性能回归风险 | 高(未知画像) | 低 | 低 |
| 与仓库 SOP 契合 | 差(换引擎=换基线) | 好 | 好(代码先行、tiny 验证、A/B 门) |

---

## 7. 推荐方案详细设计(路线 C)

### 7.1 总体拓扑与设计决策

```
        客户端(OpenAI API)
              │ SSE
      ┌───────▼────────┐
      │  PD Router(新,薄) │  健康检查 / (后期)成本模型路由
      └───────┬────────┘
              │ /v1/chat/completions
   ┌──────────▼───────────┐          KV + 元数据           ┌──────────────────┐
   │ P 节点(机 A)          │ ───────────────────────────▶ │ D 节点(机 B)      │
   │  api_server + 协调线程  │   ① admit/reserve(RPC)      │  api_server + 协调 │
   │  LLMEngine TP2(不动)  │   ② KV 块传输(通道后端)      │  LLMEngine TP2(不动)│
   │  rank0 ⇄ shm+Event ⇄ rank1 │  rank0→rank0, rank1→rank1  │  decode-only      │
   │  prefill-only          │   ③ done 回执(RPC)           │  eager+bmm_dense  │
   └────────────────────────┘                               └──────────────────┘
        通道后端:tcp_pinned(首发) / mooncake_tcp / mooncake_rdma
```

**设计决策(DD)**:
- **DD1 不做跨机 TP**:每节点内部维持现有 TP2 + MCCL + shm 命令通道原样不动(agent 报告确认三处同机假设全部保留即可);跨机只有两条独立通道:控制 RPC(小消息)与 KV 数据通道(rank 对 rank:P-rank0→D-rank0,P-rank1→D-rank1,kv 头分片天然对齐)。
- **DD2 预留-再传输**(对标 SGLang early 策略):D 节点先按块数预留本地块、返回自己的块表,P 节点按 `(src_block, dst_block)` 映射灌入——杜绝"KV 到了没地方放"。nanoPD 两端共享 BlockSpaceManager 的设计跨机不可用,这是必须换的(它自己文档也承认)。
- **DD3 P 节点持有客户端 SSE 连接端到端**:P 出 first token 后中继 D 的 token 流(P 有 tokenizer 做 IncrementalDetokenize;D 只回 token id,不需要 chat template)。Router 保持纯 L7 转发。
- **DD4 降级路径**:P 节点保留 collocated 完整能力(类似 vLLM `kv_role=both`),D 节点不可达/传输超时时 P 本地续算,请求不失败,指标打上 fallback 标记。

### 7.2 组件与职责

| 组件 | 部署 | 职责 |
|---|---|---|
| PD Router | 新进程(FastAPI,~200 行) | OpenAI 端点、双节点健康检查、(M4 后)按 prompt 长度/成本路由、fallback |
| P 协调线程(PD-Coordinator-P) | 并入 P 节点 api_server 进程 | 接收 router 请求、驱动 prefill、向 D 发 admit、编排传输、SSE 中继 |
| D 协调线程(PD-Coordinator-D) | 并入 D 节点 api_server 进程 | 块预留、接收元数据、确认传输完成、向引擎注入序列、token 流回传 |
| KV 通道后端 | 两节点对等 | `register / send_blocks / completion-event` 三接口,块粒度 |
| 引擎角色开关 | 两节点 | `Config.role ∈ {collocated, prefill_only, decode_only}` |

### 7.3 KV 导出/导入契约(新接口)

```python
# ---- P 侧导出(nanovllm/engine/block_manager.py / model_runner.py 新增)----
def pd_export(seq) -> PDKVHandle:
    # PDKVHandle = {
    #   req_id, num_tokens,           # 已算 token 数
    #   block_table: list[int],       # P 节点块号(rank0/1 相同,现有调度已保证)
    #   block_bytes: 12*MiB,          # 每 rank 每块;布局 [2, 48, 256, 2, 128] bf16 连续
    #   kv_ptr_per_rank,              # rank r 的 kv_cache[:, :, block] 切片基地址(连续 12 MiB)
    #   first_token, sampling_params, eos_ids, max_tokens,
    # }

# ---- D 侧导入 ----
def pd_reserve(num_blocks: int) -> list[int]: ...      # BlockManager 新入口:预留 n 块,返回本地块表
def pd_import(handle, dst_block_table, src_dst_map): ...
    # 物理写入:对每个 (src_b, dst_b):D-rank r 的 kv_cache[:, :, dst_b] ← 通道收到的一半分片
    # 写完置 Sequence: block_table=dst_block_table, num_computed_tokens=num_tokens,
    #   last_token=first_token → 直接进 running(prepare_decode 天然兼容,model_runner.py:572-587)
    # 前缀缓存:导入块 hash 不可复用 → 该序列关闭 hash_blocks(或同步 hash;MVP 先关闭)
```

要点:
- 现有 KV 池是单张量连续池,`kv_cache[:, :, b]` 即 12 MiB 连续切片,**导出无需重排**;
- 传输按 rank 对 rank(每 rank 48 KiB/token),双 rank 并行,通道带宽翻倍利用;
- 校验:每请求对首尾块各算 xxhash(复用 BlockManager 的 hash 函数)随元数据发送,D 侧抽查。

### 7.4 控制协议(对标 SGLang bootstrap,简化版)

```
P 协调线程                          D 协调线程
  │ ① POST /internal/pd/admit          │
  │   {req_id, num_blocks, meta}  ───▶ │ pd_reserve(n) → dst_block_table
  │ ◀─── 200 {dst_block_table, sess}   │ (预留超时 300s;D 健康由心跳保证)
  │ ② 通道传输(块粒度,双 rank 并行,     │
  │    transfer_stream 异步,不阻塞      │ (接收侧同样异步 H2D)
  │    prefill 继续算后续请求)            │
  │ ③ POST /internal/pd/kv_done        │
  │   {req_id, checksums}         ───▶ │ pd_import → 注入 running
  │ ◀─── 200                           │ token 流开始 → P 中继给客户端
  │ ④ (客户端 cancel / 节点死亡)         │
  │    cancel 双向传播;预留 300s 超时自动释放│
```

失败矩阵:传输中 P 死 → D 预留超时释放;传输中 D 死 → P 检测后走 DD4 本地续算;D OOM → admit 阶段就拒绝(P 收到后本地 decode,fallback 标记);传输超时/校验失败 → 重传一次,再失败 fallback。

### 7.5 传输通道抽象

```python
class KVChannel(ABC):
    def register(self, kv_cache: Tensor, block_bytes: int) -> Handle: ...
    def send(self, peer, src: Handle, src_ids: list[int],
             dst: Handle, dst_ids: list[int]) -> Event: ...
# 后端:
#   tcp_pinned : extract→pinned(transfer_stream, non_blocking)→socket 流式→对端 pinned→H2D
#                (nanoPD kv_transfer.py 的 extract/load 两函数直接改造复用,中间换 socket)
#   mooncake_tcp / mooncake_rdma : mooncake-transfer-engine-musa wheel;register→transfer;
#                RDMA 前提 mthreads-peermem(M0 实测)
```

`tcp_pinned` 首发的原因:零新依赖、torch_musa pinned memory 有代码级确认(`PinnedMemoryAllocator.h`)、行为可全量单测;Mooncake 后端在**同一接口**后面接入,CI 用 tcp_pinned,压测按后端分臂。

### 7.6 引擎改动点清单(文件级)

| # | 改动 | 位置 | 规模 |
|---|---|---|---|
| 1 | `Config.role` 字段 + 校验 | `nanovllm/engine/config.py` | 小 |
| 2 | prefill-only:调度跳过 decode 段、prefill 完成即产出 first token 并移交协调线程 | `scheduler.py:152-233`、`llm_engine.py:100-155` | 中 |
| 3 | decode-only:新增 `admit_remote(handle)` 注入接口绕过本地 add/allocate;`postprocess` 完成语义保持 | `scheduler.py`、`sequence.py`(num_prompt_tokens 语义=远端已填长度) | 中 |
| 4 | BlockManager:`pd_reserve`/指定块灌入;导入序列关 hash | `engine/block_manager.py:39-145` | 小 |
| 5 | KV 句柄导出(连续切片视图 + 基地址) | `model_runner.py:503-514` | 小 |
| 6 | 协调线程 ×2 + 控制 RPC(FastAPI 内部端点) | `entrypoints/openai/api_server.py` 旁路 | 中 |
| 7 | KVChannel 抽象 + tcp_pinned 后端(+ buffer pool,顺手修 nanoPD 的无池化短板) | 新文件 `nanovllm/pd/` | 中 |
| 8 | SSE 中继(first token → D token 流拼接 + usage/metrics 拼接) | api_server | 小-中 |
| 9 | PD Router | 新脚本 `scripts/pd_router.py` | 小 |
| 10 | admission 放宽:P 节点不计 max_tokens 预留 | `llm_engine.py:75-95` | 小 |

不动的东西:TP2/MCCL/shm 命令通道/runner.call、MoE 双后端(自动按 is_prefill 分流,PD 下反而更干净:P 天然 grouped_mm,D 天然 bmm_dense)、triton_paged、benchmark_online.py。

### 7.7 路由与成本模型(后期,M5)

nanoPD 公式照抄,四参数 + bandwidth 在 S4000 重测(profiler 的 prefill/decode/interference/带宽四个 profile 直接可用,只换 torch.cuda→torch.musa)。MVP 阶段固定全量走 PD + prompt 长度硬阈值(如 L≥512 才走 PD,短请求 P 本地 collocated——传输成本与收益都是 L 的线性函数,nanoPD 的闭式判据 γ/transfer_rate > batch_thresh/system_load 在 1P1D 下简化为"短请求不值得搬")。

### 7.8 可观测与验收口径

- 沿用 step 级 JSONL + `/metrics`;新增 PD 指标:`pd_transfer_bytes_total`、`pd_transfer_seconds`、`pd_handoff_latency`、`pd_fallback_total`、D 节点 `prefill_recompute_count`(必须恒 0,这是"KV 真的传过去了"的完整性判据,对齐 P13 的拒绝率完整性要求精神)。
- 压测:`benchmark_online.py` 打 Router,同冻结负载,collocated(现状生产)vs PD 交错 ≥5 次、中位数、CV≤5% 门,双 SLO + goodput 全量输出。

---

## 8. 难度评估(分模块)

| 模块 | 难度 | 预估 | 依据 |
|---|---|---|---|
| KV 导出/导入原语 | ★★☆ | 2–3 天 | 布局连续无重排;难点在"指定块灌入"绕开现有顺序分配(接口新,非算法难) |
| 角色拆分(scheduler/config) | ★★★ | 3–4 天 | 语义改动面广(准入/完成/抢占),但有现成 mode 字段与单测护城河(p8_scheduler_gaps) |
| 控制协议 + 协调线程 | ★★☆ | 3–4 天 | SGLang bootstrap 协议可抄;纯工程 |
| tcp_pinned 通道 | ★★☆ | 2–3 天 | nanoPD 骨架 + socket;风险在 torch_musa pinned 异步语义需 tiny 验证 |
| Mooncake TE 后端 | ★★★(条件) | 2–3 天 | wheel 兼容性未知;接口已隔离,失败可退 tcp_pinned |
| mooncake_rdma | ★★★★(未知) | 不可估 | peermem 未证实;单独 gate |
| SSE 中继 + Router | ★★☆ | 2 天 | 纯工程 |
| 30B 跨机 bring-up | ★★★ | 3–5 天 | SOP 成熟但跨机首跑坑位多(防火墙/端口/shm 清理/双机协同) |
| A/B 判定 | ★★☆ | 2–3 天 | 协议现成,重在预注册判据 |
| **合计(MVP→生产)** | **3.5/5** | **4–6 周**(含 0.6B MVP 1.5–2 周) | 不确定性集中在传输层与网络,不在引擎逻辑 |

---

## 9. 分阶段实施计划(遵守仓库 SOP:代码先行、tiny 验证在 30B 加载前完成、一次一个 GPU 任务、45min 问题盒、到点交接清场)

| 阶段 | 内容 | 验收 gate | GPU 预算 |
|---|---|---|---|
| **M0 探针**(1 天) | ① 两机 `iperf3` 互测带宽;② `lspci`/HCA/`peermem` 模块/mthreads-gmi 拓扑;③ S4000 试装 `mooncake-transfer-engine-musa` + `get_local_topology()`;④ clone vllm-musa 读 `mooncake_compat.py`,测试机跑通其单机 1P1D 示例(对照证据);⑤ 现网 ITL 尾部量测(§1.2 Step 0) | 《M0 探针纪要》:可用通道排序 + go/no-go;**若 ITL 尾部 <2× 且无长上下文诉求,整个项目降级为"记录归档"** | ~0(系统/网络层,1 次单机短窗口) |
| **M1 KV 原语**(2–3 天) | §7.3 的 4/5 号改动;0.6B 单卡:填 KV→导出→重灌→续 decode logits 与原路径**全等** | 全等 rc=0(六配置全等式标准);TP2 双 rank 分片导出/灌入全等 | 0.6B,单机 |
| **M2 单机 1P1D**(3–4 天) | 1/2/3/6/7/8/9 号改动,通道用 loopback;0.6B:P=卡0,D=卡1(或双 TP2 同机走 shm 通道) | benchmark_online 打通 PD 全链路;`prefill_recompute_count==0`;TTFT/TPOT 完整 | 0.6B,单机 |
| **M3 跨机 bring-up**(3–5 天) | 两机部署,tcp_pinned 实测跨机带宽;Mooncake TCP 后端同接口对比;0.6B 跨机 PD;30B TP2×2 节点首通(加载窗口只做测量) | 跨机带宽记录(GB/s);30B PD 首请求端到端;校验和抽查通过 | 0.6B + 1 次 30B 窗口 |
| **M4 生产 A/B**(4–6 天) | 生产栈两臂:collocated(现状)vs PD;P=TP2+grouped_mm+chunk256,D=TP2+bmm_dense+eager+seqs≤64;交错 ≥5、CV≤5% | 预注册判据(建议:TTFT p50/p95 改善 ≥10% 且 TPOT p95 恶化 ≤5%,goodput 不降;不达标则按 SOP 记负结果入 ISSUES) | 30B,多窗口 |
| **M5 优化**(可选 1–2 周) | layer-wise 传输重叠(白皮书手法:chunk-256 内逐层边算边传);成本模型自适应路由(S4000 重测参数);Mooncake Store 前缀复用;RDMA(若 M0 证实 peermem);KV FP8(联动 I20,单独立项) | 各自单独 A/B 门 | 按项另计 |

并行度说明:两台机器是两个 GPU 资源,P/D 同跑不违反"一次一个 GPU 任务"(该规则按机器计);M0–M2 期间服务器 A 仍须让位给 P13 在跑实验(P13 当前 PENDING,M1/M2 用 0.6B 短窗口可穿插)。

---

## 10. 风险登记册

| # | 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|---|
| R1 | 两机间网络 <10 Gbps(云租户互通) | 中 | TCP 路线 4K+ 请求 TTFT 增益被传输吃掉 | M0 实测前置;按 prompt 长度路由只走长请求;RDMA gate |
| R2 | S4000 无 mthreads-peermem | 中 | RDMA 不可用,锁死 TCP 上限 | tcp_pinned 保底已可交付功能;白皮书证明 MUSA+RDMA 链路存在(S5000),可向摩尔线程工单确认 S4000 支持矩阵 |
| R3 | mooncake MUSA wheel 与 S4000 不兼容 | 低-中 | 退路线 B(纯自研通道) | 接口隔离(§7.5),C 退化为 B,不返工 |
| R4 | torch_musa pinned 异步/事件语义与 CUDA 有差 | 低-中 | 传输正确性 | M1 全等单测 + M2 双流重叠 tiny 验证(对齐 musa-pitfalls 的快照三件套纪律) |
| R5 | D 节点 eager 容量受限,PD 吞吐收益≈0 | 中 | 项目价值打折 | 这正是 M4 A/B 预注册判据要回答的问题;负结果按 SOP 原样入账 |
| R6 | 短请求(≤512)PD 净亏损(传输>收益) | 高(确定) | 无 | 硬阈值路由,nanoPD 成本模型后期接管 |
| R7 | I29:任何在 D 节点重启图加速的念头 | — | 红线 | 明确:D 节点永远 eager,直到 I29 复活条件满足 |
| R8 | 跨机首跑协同坑(防火墙/端口/两机 shm 清理节奏) | 中 | 时间损耗 | M3 预留缓冲;两机各自按 SOP 清场;45min 问题盒纪律 |
| R9 | 与 P13/I20 地盘冲突(PD 若顺带做 KV 压缩) | — | 流程红线 | PD 初版 KV 恒 bf16;KV FP8 显式划给 I20 |

---

## 11. 参考链接

**vLLM**: [disagg_prefill 文档](https://docs.vllm.ai/en/latest/features/disagg_prefill.html) · [MooncakeConnector](https://docs.vllm.ai/en/stable/features/mooncake_connector_usage) · [NixlConnector](https://docs.vllm.ai/en/stable/features/nixl_connector_usage) · [vllm#11490](https://github.com/vllm-project/vllm/issues/11490) · [production-stack disagg](https://github.com/vllm-project/production-stack/blob/main/docs/source/use_cases/disaggregated-prefill.rst) · [LMCache MP P/D](https://docs.lmcache.ai/mp/disaggregated_prefill.html) · [LMCache NIXL 博客](https://blog.lmcache.ai/en/2025/04/11/shaping-nixl-based-pd-disaggregation-in-vllm-v1) · [AWS HyperPod DPD](https://aws.amazon.com/blogs/machine-learning/disaggregated-prefill-and-decode-for-llm-inference-on-sagemaker-hyperpod)

**SGLang / Mooncake**: [SGLang PD disagg](https://docs.sglang.io/advanced_features/pd_disaggregation.html) · [sglang#15674](https://github.com/sgl-project/sglang/issues/15674) · [Mooncake 仓库](https://github.com/kvcache-ai/Mooncake) · [build.md(USE_MUSA/peermem)](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/getting_started/build.md) · [Mooncake#1883(MUSA 时间线)](https://github.com/kvcache-ai/Mooncake/issues/1883) · [mooncake-transfer-engine-musa(PyPI)](https://pypi.org/project/mooncake-transfer-engine-musa/) · [Mooncake 论文 OSDI'24](https://arxiv.org/abs/2407.00079) · [DistServe 论文](https://arxiv.org/abs/2401.09670)

**NVIDIA**: [NIXL 博客](https://developer.nvidia.com/blog/enhancing-distributed-inference-performance-with-the-nvidia-inference-transfer-library) · [nixl 仓库](https://github.com/ai-dynamo/nixl) · [Dynamo KV-aware routing](https://docs.nvidia.com/dynamo/cli/kv-aware-routing/overview) · [llm-d P/D 指南(含 Moore Threads 行)](https://llm-d.ai/docs/well-lit-paths/foundations/pd-disaggregation)

**摩尔线程**: [vllm-musa 仓库](https://github.com/MooreThreads/vllm-musa) · [vllm-musa disagg 示例](https://github.com/MooreThreads/vllm-musa/blob/v0.28.0-dev/example/disaggregated_serving/README.md) · [torch_musa](https://github.com/MooreThreads/torch_musa) · [MUSA SDK 页(MCCL 多机声明)](https://www.mthreads.com/product/MUSASDK) · [KUAE](https://www.mthreads.com/product/KUAE) · [PD 分离白皮书(S5000+H200)](https://developer-hscdn.mthreads.com/public/PD%E5%88%86%E7%A6%BB%E5%BC%82%E6%9E%84%E7%AE%97%E5%8A%9B%E6%B7%B7%E9%83%A8%E5%AE%9E%E8%B7%B5%E6%8A%80%E6%9C%AF%E7%99%AE%E7%9A%AE%E4%B9%A6.pdf)(调研时已下载核读,临时件 `/tmp/mt_pd_whitepaper.pdf`) · [白皮书论坛帖](https://developer.mthreads.com/forum/d/8-pd-fen-chi-yi-gou-suan-li-hun-bu-shi-jian-ji-zhu-bai-pi-shu) · [至顶网报道](https://www.zhiding.cn/ai/2026/0821/3197073.shtml) · [科技日报·趋境科技合作](https://www.stdaily.com/web/gdxw/2026-09/05/content_575716.html) · [摩尔线程博客中心(MUSA Wheel 发布文)](https://blog.mthreads.com) · [vLLM-MUSA S4000 快速上手](https://docs.mthreads.com/vllm-musa/vllm-musa-doc-online/S4000/quick_start) · [SGLang-MUSA 文档](https://docs.mthreads.com/sglang-musa-m1000/version-20260830/sglang-musa-m1000-doc-online/quick_start) · [sglang#16565(MUSA 路线图)](https://github.com/sgl-project/sglang/issues/16565)

**工程参考**: [Sarathi-Serve / KV Transfer Math(dev.to)](https://dev.to/wolfsea2357/prefilldecode-disaggregation-in-vllm-the-kv-transfer-math-that-decides-if-it-pays-off-13j8) · [阿里云 eRDMA 双机部署指南](https://help.aliyun.com/zh/ecs/user-guide/qwen3-235b-a22b-model-of-dual-machine-deployment-based-on-erdma) · [NVIDIA 论坛双卡 LMCache P/D 教程](https://forums.developer.nvidia.com/t/disaggregated-prefill-decode-split-using-lmcache-dual-nvidia-rtx-pro-6000-blackwell/360634) · [dstack P/D 指南](https://dstack.ai/blog/pd-disaggregation) · [TaiChi 混合调度](https://arxiv.org/abs/2505.19050)

**本地代码**: nanoPD 关键文件 `nanoPD/workers/kv_transfer.py`(82 行,传输切面)、`router/central_scheduler.py`(状态机)、`cost_model/`(公式);本仓库集成面 `nanovllm/engine/model_runner.py:572-587`(prepare_decode 天然兼容外来序列)、`engine/scheduler.py:152-233`、`engine/block_manager.py`。

## P14 补充:单机侧 M0 探针结果(2026-10-30,只读,未安装任何东西)

- **NIC**:`lspci` 见 ConnectX-6(MT28908);`/sys/class/infiniband` 有 `mlx5_0`、`mlx5_1`、`mlx5_bond_0`;容器内网口仅 `eth0`(veth),RDMA 设备可见。
- **GPUDirect RDMA 前置**:`lsmod` 已加载 `mt_peermem`(依赖 `ib_uverbs`、被 `mtgpu` 使用)。
- **拓扑**(`mthreads-gmi topo -m`):GPU0↔GPU1 为 MPB;两卡与 NIC0/NIC1 均为 NODE(同 NUMA 0),NIC2 为 SYS。
- **mooncake**:`pip download mooncake-transfer-engine-musa --no-deps` 成功取得 `0.3.13.post1` cp310 manylinux wheel(33 MB);**未安装**(红线:不改 site-packages;安装验证留给有第二台机器的下一轮)。
- **仍缺**:两机带宽/RDMA 实测(无第二台机器)。
