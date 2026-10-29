# P13 附:业界参考方案调研(联网调研报告,2026-10-29)

> 目的:为 [PROMPT-P13.md](./PROMPT-P13.md) 的四个攻坚项(I30/I33/I8/I20)与一个厂商项(I29)寻找业界参考方案与可移植做法。
> 方法:公开资料(GitHub、官方文档、论文、技术博客);**全部为【文献】级证据,未在本机复现**——沿用 [P6-musa-graph-survey.md](git 历史 `14d9954`)的标注纪律:文献结论不得当本机结论引用。本机已验证事实以 ISSUES/HANDOFF 为准。

## 0. 速览:问题 → 业界方案 → 一句话结论

| 我们的问题 | 业界对应物 | 核心做法 | 对我们的结论 |
|---|---|---|---|
| **I33 桶 padding 税**(在线慢 5.9–14.7%) | vLLM / SGLang 的图桶策略 | **密桶梯子**(vLLM 默认步长 8)+ **padding 时回退 eager**(SGLang 有现成 flag) | 我们的桶太稀疏;业界把 padding 税当标准问题管理,而不是接受它 |
| **I33 分段图本身** | vLLM V1 piecewise CUDA graph、SGLang Piecewise/Breakable CUDA Graph | 在 attention/MoE-dispatch 等"图不安全算子"处切段,段内捕获 | 我们独立收敛到了同一条路线;设计被业界大规模验证 |
| **I8 长上下文 TTFT 崩** | Sarathi-Serve(OSDI'24)、vLLM V1 token budget、SGLang schedule-conservativeness | chunked prefill + 每步 token 预算 + KV 感知准入 | 我们缺的是"**chunk 大小当负载参数调**"+ 端到端复测,不是缺机制 |
| **I20 KV 量化** | vLLM(仅 FP8)、TensorRT-LLM(FP8+INT8)、IPEX-LLM/OpenVINO(无 FP8 硬件上 INT8/INT4 KV)、KIVI(学术 2-bit) | INT8 存储偏移尺度 + 读时反量化,不需要 FP8 算力单元 | 路线成熟可抄;但 I8 的崩点在调度不在 KV,**先修调度再谈 KV** |
| **I29 图内 MCCL 楔死** | vllm-musa 官方(S4000 文档默认开图) | 厂商 MoE 代码是"捕获感知"的 | 我们有最小复现,该打包成厂商工单;这是唯一我们自己修不了的项 |
| **I30 口径复核** | (方法论问题,无外部方案) | 同 session A/B | 已在任务书里解决设计,无需外部参考 |

---

## 1. I33:业界怎么管住"桶 + padding"的税

### 1.1 vLLM 的捕获梯子是密的

vLLM 官方文档(`vllm/config`,`CompilationConfig.cudagraph_capture_sizes`)给出的默认梯子:

> `[1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 512, 16)) + list(range(512, 2048, 32))`

即 **8 起步长 8**:`1,2,4,8,16,24,32,40,48,56,64,…`。运行时 batch 向上取最近桶;超过最大桶则 eager。
出处:[vLLM config 文档](https://docs.vllm.ai/en/v0.18.2/api/vllm/config)、[CUDA Graphs 设计文档](https://docs.vllm.ai/en/v0.18.0/design/cuda_graphs)、[aiengineeringfromscratch 图解](https://aiengineeringfromscratch.com)。

**逐 batch 对比**(我们实测并发带 14–30,P12 三档 max_concurrency 14/24–28/21–23/27–30):

| batch | 我们的桶 → cap | pad 比 | vLLM 梯子 → cap | pad 比 |
|---|---|---|---|---|
| 14 | 16 | 1.14 | 16 | 1.14 |
| 17 | **32** | **1.88** | 24 | 1.41 |
| 21 | **32** | **1.52** | 24 | 1.14 |
| 25 | 32 | 1.28 | 32 | 1.28 |
| 30 | 32 | 1.07 | 32 | 1.07 |

我们最痛的 17–23 区间恰好对应 P12 回退最大的 0.9 档(+14.7%,并发 21–23,pad 1.4–1.5×)——相关性成立但不是证明,判别实验仍是任务书的 **b16 vs b17**。

### 1.2 SGLang 有现成的"padding 就回退 eager"开关

SGLang server args:

> `--disable-cuda-graph-padding` — "Disable cuda graph when padding is needed. Still uses cuda graph when padding is not needed."

即**桶精确匹配才 replay,需要 padding 的 batch 走 eager**——这就是任务书 I33 修复选项②的业界同款,已发布、有文档。
出处:[SGLang server arguments](https://sgl-project.github.io/advanced_features/server_arguments.html)、[RFC #29184](https://github.com/sgl-project/sglang/issues/29184)(该 flag 正在被重构合并)。

### 1.3 分段/可断图是业界主流,我们押对了路线

- **vLLM V1 piecewise CUDA graph**:torch.compile 分段,在 attention 处拆图,attention eager 跑在图段之间([vLLM 官方文档](https://docs.vllm.ai)、[vLLM 论坛设计讨论](https://discuss.vllm.ai/t/questions-on-piecewise-torch-compile-design/589))。
- **SGLang Piecewise CUDA Graph**:按注册 split point 切图(每个 MoE 层一个),token-bucket 捕获([文档](https://sgl-project.github.io/advanced_features/piecewise_cuda_graph.html)、[roadmap issue #11490](https://github.com/sgl-project/sglang/issues/11490))。
- **SGLang Breakable CUDA Graph(BCG,2026 起为默认)**:不再依赖 torch.compile,原生捕获 + 图断点;断点插在图不安全算子处(**attention 后端、MoE dispatch、custom all-reduce、top-k**);无断点时开销与整图几乎一致([LMSYS 博客](https://www.lmsys.org/blog/2026-08-17-advanced-cuda-graph)、[官方文档](https://sgl-project.github.io/advanced_features/breakable_cuda_graph.html)、[社区解析](https://yichizhang.dev/AI-Infra-Visualized/en/lessons/breakable-cuda-graph))。
  - 我们 git 历史里的 P6 图谱调查 §D 已记录其 API(`@eager_on_graph` / `break_graph()`)可照抄——与 P10 的 `defer_reduce` 手法同构。

**含义**:分段图路线本身没有错,业界(包括 Meta 合作)已把它做成默认;我们 P12 的 FAIL 归因于**桶稀疏 + 无回退**,不是路线错误。这与任务书"复活门(1.0 档须优于 eager ≥3%)"组合起来就是正确姿态:修桶值得做一轮,但期望要按 P11 的边界数据校准。

### 1.4 必须诚实校准的期望:P11 边界数据给修复设了天花板

P11(同 session,CV≤0.1%)在**桶边界**(pad=1)实测:segment b8 **+0.5%**、b32 **+3.8%**。也就是说,即使密桶把 padding 归零,30B 上的期望收益是"**回到 ≈0~−4%**",不是转正——因为 b8 起设备已进饱和区,且每层 2 次 eager AR 的串行化把 launch 收益吃掉(这与业界"图收益随 batch 增大而缩小"的普遍经验一致,只是我们这边在 b≥8 时提前转为净负)。

**结论链**:密桶 + 回退能把 +5.9~14.7% 的在线回退**修到 ≈0%,但修不成正收益**;要拿到 b≥8 的正收益,必须把 AR 也放进图,即回到 I29。**修复(I33)的价值是把段图从"在线有害"变成"在线无害的单流开关",并顺手补上离线门的盲区**——值一轮小实验,不值得再赌转正。

### 1.5 落地顺序建议(并入 PROMPT-P13 P1;⚠️ 先看 §1.6 的驱动上限)

1. **先做回退 flag**(改动最小、零显存):`cap/bs > 阈值(建议 1.15)` 时步回退 eager;阈值做成 env 可调,SGLang 的语义=精确匹配才 replay,可直接对齐。
2. **再做密桶**(step 4 或 8,到 64):768 图 → ~1300–1700 图/rank;捕获时间按 5.2 s 线性外推 ~10 s 级,图池增量必须实测(I32-② 的 OOM 教训:KV 先填 0.9 包络)。
3. **门**:0.6B b6–b32 全点扫描(含 b16/b17 判别)→ 30B 单轨步时 → 30B 在线 A/B(复活门:1.0 档 ≥+3%)。

### 1.6 ⚠️ 厂商硬上限:MUSA 驱动限制每进程存活图可执行文件 ~2048(torchada 披露)

torchada(摩尔官方的 CUDA API 兼容层)README 的 CUDA Graphs 章节披露([torchada README](https://raw.githubusercontent.com/MooreThreads/torchada)):

> "The MUSA driver caps the number of live graph *executables* per process (~2048)"
> "piecewise CUDA graphs instantiate `capture_sizes * num_layers` of them, so models deeper than roughly 40 layers used to exceed the cap and could not use piecewise graphs at all."

torchada 的解法是保留模板 + LRU 轮换(默认上限 1900,回放被逐出的图时从模板重新实例化,~0.3 ms/次;`TORCHADA_GRAPH_ROTATION=1` 默认开):

> "torchada keeps every captured *template* alive, LRU-caps the live executables (1900 by default), and re-instantiates an evicted graph's executable from its template on the next replay — which costs about 0.3 ms and needs no forward re-run"

**对我们的直接推论**:

- 我们 30B = 48 层 × 2 段 → **桶数硬上限 ≈ 1900/96 ≈ 19–21 桶**(在默认 1900 cap 下)。768 图(8 桶)安全;step-4 到 64(18 桶)= 1728 图**可行但贴近上限**;I32-② 的 maxseq-512 35 桶 3360 图 OOM 除显存外,可能还叠加了这个驱动上限(1632/3360 失败点与 2048 同量级)——**值得在工单里问厂商求证**。
- 若做密桶,先在 tiny/0.6B 上验证桶数×层数×2 与驱动上限的关系;或确认服务器 torch_musa 2.11 是否已内置类似轮换(torchada 是独立兼容层,我们直连 torch_musa,不一定享受该修复)。
- **同一 README 还披露图调试转储**:`TORCHADA_CUDA_GRAPH_DEBUG_DUMP_PATH` → 每次捕获自动 `enable_debug_mode()` + `debug_dump()` 写时间戳 `.dot` 文件——**这正是我们缺的 MUSA 图级可观测性**(E6 痛点、I29 诊断慢的根因)。探针实验可直接用它 dump 段图结构对照。

---

## 2. I8/I26:长上下文 TTFT——机制我们不缺,缺调参与复测

### 2.1 Sarathi-Serve(OSDI'24):我们调度器的理论出处

- chunked prefill 把 prefill 切成与 decode 混批的小块,"stall-free"让 decode 几乎不被 prefill 顿住([arXiv 2403.02310](https://arxiv.org/html/2403.02310v1)、[OSDI'24 页面](https://www.usenix.org/conference/osdi24/presentation/agrawal)、[笔记](https://github.com/junuxyz/mlsys-notes/blob/main/notes/sarathi-serve.md))。
- 论文的核心工程结论之一:**chunk 大小是可调参数**,按模型与硬件的 SM 占用推最优值;太小→prefill 拉长(TTFT 差),太大→decode 顿(TPOT 差)。这正是我们 P9/P10 的两难实测(chunk-2048 TTFT −69~73% 但短 prompt TPOT +5~9.5%)的学术镜像。
- 2026 后续:MLSys'26 "From Tokens to Layers" 把 stall-free 细到 layer 级([页面](https://mlsys.org/virtual/2026/oral/3732))——超出我们需求,记录备查。

### 2.2 vLLM V1:每步 token 预算 + 一次一个 chunked prefill

- V1 默认开启 chunked prefill,每步 token 预算 = `max_num_batched_tokens`(默认 8192),**预算越大 TTFT 越好**;调度器每步只推进一个 chunked prefill,预算有剩余才并第二个([vLLM 优化指南](https://docs.vllm.ai/en/stable/configuration/optimization)、[论坛:V1 调度策略](https://discuss.vllm.ai/t/v1-chunked-prefill-scheduling-policy-how-prefill-would-be-scheduled/146)、[深度解析](https://audreywongkg.medium.com/understanding-vllm-scheduling-token-budgets-chunked-prefill-and-policies-2c879e3980e3))。
- 对 1k prompt:vLLM 会一口气给 1k token 一整步预算(除非并发挤占),所以它的 1k TTFT 不被自己的 chunk 粒度卡住;我们的 chunk-256 把 1k 拆成 4 步排队,这是 I8 的 TTFT 崩的主因之一。
- 另一条社区共识:chunked prefill 不减少总计算量,它只决定"这些计算何时插队"([X 上的概括](https://x.com/akshay_pachaar/status/2107112561867116681)、[Spheron 解析](https://www.spheron.network/blog/chunked-prefill-explained-how-it-smooths-ttft-under-load-202))——与 I8 结论"崩在排队不是算力"完全一致。

### 2.3 SGLang:准入激进度是显式旋钮

`--schedule-conservativeness`(默认 1.0):控制依据 KV 余量放新请求进来的激进程度,<1 更保守。这与我们 I26 的 429/HOL-skip 是同一职责的两种实现;SGLang 还有 `--max-running-requests` 硬顶。([server args 文档](https://sgl-project.github.io/advanced_features/server_arguments.html))

### 2.4 对任务书 P2 的具体强化

1. **加 chunk 档位臂**:256(生产默认)vs 512/1024。假设:P9 已证 1k 档 chunk-2048 TTFT 1.036 s 达标;本轮在"准入修复后"的栈上复测,预期 512 或 1024 是 1k 负载的甜点(2048 可能导致短 TPOT 档塌,我们已测过)。**不翻默认**,只给"1k 场景推荐配置"。
2. **拒绝率完整性**(任务书已写):对齐 SGLang 的做法,把 `--schedule-conservativeness` 类比的激进度也当作报告维度——同一负载下"429 拒绝率 × SLO 达标率"一起看,否则不是容量结论而是准入阈值结论。
3. **分解报告**:TTFT p95 拆"排队时间 + prefill 计算时间"(Sarathi 论文用的分解法),才能回答"还差在哪"。

---

## 3. I20:KV 量化——路线成熟,但先确认它不是当前瓶颈

### 3.1 生态现状

| 框架 | KV 量化支持 | 说明 |
|---|---|---|
| vLLM | **仅 FP8**(e4m3/e5m2),`--kv-cache-dtype fp8` | INT8 只有 [feature request #33480](https://github.com/vllm-project/vllm/issues/33480);官方博客:[The State of FP8 KV-Cache](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html)、[文档](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache) |
| TensorRT-LLM | FP8 + INT8 | [对比文](https://blog.squeezebits.com/vllm-vs-tensorrtllm-8-kv-cache-quantization-35079) |
| **IPEX-LLM**(Intel) | **INT8/INT4,无需 FP8 硬件** | [intel/ipex-llm](https://github.com/intel/ipex-llm)——与我们处境最像的参考(Intel GPU 无 FP8 单元,INT8 存 + 反量化算) |
| OpenVINO(2026.2) | INT8 默认,新增 INT4 | [发布文](https://medium.com/openvino-toolkit/int4-kv-cache-compression-for-llm-inference-on-intel-gpu-new-in-openvino-2026-2-d71d03c27897) |
| 学术 | KIVI 2-bit(K per-channel / V per-token,免调参,[arXiv 2402.02750](https://arxiv.org/abs/2402.02750));INT8 KV GPU 实现 4×([arXiv 2601.04719](https://arxiv.org/html/2601.04719v1)) | 数值门设计的参照 |

**可移植路线**(无 FP8 硬件同款):INT8 存储 + per-layer/per-head scale + 读时反量化;收益 = KV 容量 ×2 + decode attention 的 KV 读带宽减半。可行性探针清单:① `torch_musa` 的 int8↔bf16 转换带宽;② 把反量化塞进 `paged_attention.py` Triton kernel 的代价;③ I32 的 KV 预留改造顺路做。

### 3.2 必须先泼的冷水

I8 的实测结论是**崩在 TTFT/排队,不是 KV 容量**(三档 SLO 0% 时 KV 没满;384/384 全部完成)。所以 KV 量化解决的是"调度修好之后的天花板"(更高并发),不是 I8 本身。**顺序:先跑 P13-P2(调度/准入),看 1k 档修完后 KV 余量还剩多少,再决定 I20 立项**。这已写入任务书 P4 的前置条件。

---

## 4. I29:厂商侧的图内 MCCL——该升级成正式工单了

### 4.1 我们已有的本机事实(汇总自 ISSUES I29 + P6 图谱调查)

- 厂商**自己的文档**写明 S4000 上 MUSA Graph **默认开启**(`--enforce-eager` 才关)——[S4000 快速开始](https://docs.mthreads.com/vllm-musa/vllm-musa-doc-online/S4000/%E5%8E%86%E5%8F%B2%E7%89%88%E6%9C%AC%E4%BF%A1%E6%81%AF/quick_start)。
- 厂商 MoE 源码是**捕获感知**的(`is_current_stream_capturing()` 分支、权重预处理保 data_ptr 不变)。
- 本机探针:纯张量图含 96 次图内 AR **全绿**;引擎级 replay 间歇楔死;0.6B TP2 paged 首次 replay 3/3 楔死(最小复现)。
- `torch._grouped_mm` 不可捕获(`operation not permitted`)→ 我们的 bmm_dense 图路径选择正确(vllm-musa 的 MoE 图内路径大概率也是 dense/bmm 家族)。

### 4.2 厂商动态(2025–2026)

- vllm-musa 现主线 **v0.28.0-dev,已转 S5000 + vLLM V1 引擎 + Runner V2**([README_CN](https://raw.githubusercontent.com/MooreThreads/vllm-musa/v0.28.0-dev/README_CN.md));PyPI 插件 2025-12 更新([vllm-musa](https://pypi.org/project/vllm-musa))。S4000 有独立版本文档线(vLLM core v0.21 等)([docs.mthreads.com](https://docs.mthreads.com/vllm-musa/vllm-musa-doc-online/S4000/releasenote))。
- torch_musa 仍在活跃发版(v2.5.0 起 version 对齐 PyTorch;v2.7.0 于 2025-11 发布;release 提到 **MUSA Graph 调试函数**与 MCCL 初始化改进)([releases](https://github.com/MooreThreads/torch_musa/releases)、[新闻](https://news.futunn.com/en/post/65528016/moore-threads-releases-torch-musa-v2-7-0-further-enhancing))。
- vLLM-Omni 也接了 MUSA 平台([文档](https://docs.vllm.ai/projects/vllm-omni/en/v0.20.0/api/vllm_omni/platforms/musa))——生态在扩,说明报 bug 有接收方。

### 4.3 建议:把工单包写出来(零 GPU 成本)

工单要素(全部现成):
1. **最小复现**:`tests/p9_mccl_graph_probe.py`(分级探针)+ 0.6B TP2 paged 首次 replay 3/3 楔死 + 三种崩溃形态;对照纯张量图(含 96 图内 AR)全绿 → 界定为"引擎 replay 协议 × MCCL"。
2. **版本**:torch 2.11.0.post1 + torch_musa 2.11、MCCL 2.11、驱动版本;S4000 双卡 PCIe。
3. **问题清单**:① 引擎级 graph replay(每步 fill vars→replay→eager logits)与 MCCL 的已知限制?② MUSAGraph 有无 CUDA 级 debug/dump 手段(**torchada 已有 `TORCHADA_CUDA_GRAPH_DEBUG_DUMP_PATH`,torch_musa 直连路径是否同样具备?**);③ **每进程 ~2048 图可执行上限的准确数值与版本、torch_musa 直连路径是否有 torchada 式的模板轮换?**(我们 I32-② 的 3360 图 OOM 疑似叠加此限制);④ `torch._grouped_mm` 捕获报 `operation not permitted` 是否预期、推荐的可捕获 MoE 路径是否就是 dense bmm?⑤ torch_musa v2.7 的 MUSA Graph 修复是否覆盖(要不要升级)?
4. **姿态**:我们是自研引擎非商业客户,但 bug 报告质量高(有最小复现+探针),且官方文档声称 S4000 默认开图 → 引擎级楔死与官方姿态矛盾,值得他们收。

---

## 5. 明确不追的方向(带业界依据)

| 方向 | 不追理由 | 业界旁证 |
|---|---|---|
| Prefill/Decode 分离部署(DistServe/Splitwise/Mooncake) | 双卡 TP2 装不下单卡副本;跨机带宽不可控 | [DistServe OSDI'24] 等需要集群前提 |
| torch.compile 路径 | MUSA 侧 `maybe_compile` 为空壳,inductor 无后端 | SGLang PCG 恰好是为绕开 compile 才转向 BCG |
| EAGLE3/投机解码 | S1 NO-GO(α≈0);前置 I19 只在 b1 被解决 | — |
| fused RMSNorm 类小算子融合 | P8 实测 −0.76%,因果链已否 | 与"小 kernel 洪水"结论并存:要融合就得成套,单点无效 |

---

## 6. 汇总:并入任务书的行动

| 行动 | 来源 | 成本 |
|---|---|---|
| P1 增加"回退 flag 优先、密桶其次"的顺序 + 梯子对比表;**密桶受驱动 ~2048 图可执行上限约束(≈19–21 桶),先在 tiny 上验证** | §1.5–1.6 | 不变(~1 h) |
| P1 期望校准:修复目标=无害化,转正仍以复活门为准 | §1.4 | 0 |
| P2 增加 chunk 档位臂(512/1024)+ TTFT 分解(排队 vs 计算) | §2.4 | +15 min |
| P4 前置:IPEX-LLM 参照 + "先看 1k 修完后 KV 余量" | §3.2 | 0 |
| **新增 P5:厂商工单包**(文本工作,零 GPU) | §4.3 | ~1 h |
| 可选:升级 torch_musa v2.7 评估(升级须过回归四套 + L1–L4) | §4.2 | 风险项,默认不做 |

## 7. 来源清单

- vLLM:[config(捕获梯子)](https://docs.vllm.ai/en/v0.18.2/api/vllm/config) · [CUDA Graphs 设计](https://docs.vllm.ai/en/v0.18.0/design/cuda_graphs) · [V1 优化与 token budget](https://docs.vllm.ai/en/stable/configuration/optimization) · [V1 调度讨论](https://discuss.vllm.ai/t/v1-chunked-prefill-scheduling-policy-how-prefill-would-be-scheduled/146) · [piecewise 设计讨论](https://discuss.vllm.ai/t/questions-on-piecewise-torch-compile-design/589) · [FP8 KV 博客](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html) · [KV 量化文档](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache) · [INT8 KV feature request #33480](https://github.com/vllm-project/vllm/issues/33480)
- SGLang:[Piecewise CUDA Graph 文档](https://sgl-project.github.io/advanced_features/piecewise_cuda_graph.html) · [Breakable CUDA Graph 文档](https://sgl-project.github.io/advanced_features/breakable_cuda_graph.html) · [LMSYS 博客](https://www.lmsys.org/blog/2026-08-17-advanced-cuda-graph) · [server args(padding 开关)](https://sgl-project.github.io/advanced_features/server_arguments.html) · [roadmap #11490](https://github.com/sgl-project/sglang/issues/11490) · [RFC #23004](https://github.com/sgl-project/sglang/issues/23004) · [BCG 社区解析](https://yichizhang.dev/AI-Infra-Visualized/en/lessons/breakable-cuda-graph)
- 论文:[Sarathi-Serve OSDI'24](https://www.usenix.org/conference/osdi24/presentation/agrawal) / [arXiv](https://arxiv.org/html/2403.02310v1) · [KIVI](https://arxiv.org/abs/2402.02750) · [INT8 KV GPU](https://arxiv.org/html/2601.04719v1) · [MLSys'26 From Tokens to Layers](https://mlsys.org/virtual/2026/oral/3732)
- KV 量化生态:[IPEX-LLM](https://github.com/intel/ipex-llm) · [OpenVINO INT4 KV](https://medium.com/openvino-toolkit/int4-kv-cache-compression-for-llm-inference-on-intel-gpu-new-in-openvino-2026-2-d71d03c27897) · [vLLM vs TRT-LLM KV](https://blog.squeezebits.com/vllm-vs-tensorrtllm-8-kv-cache-quantization-35079) · [TensorMesh 综述](https://tensormesh.ai/learn/kv-cache-quantization)
- 厂商:[vllm-musa](https://github.com/MooreThreads/vllm-musa) · [v0.28.0-dev README_CN](https://raw.githubusercontent.com/MooreThreads/vllm-musa/v0.28.0-dev/README_CN.md) · [torch_musa releases](https://github.com/MooreThreads/torch_musa/releases) · [**torchada README(图可执行上限 + 图调试转储)**](https://raw.githubusercontent.com/MooreThreads/torchada) · [S4000 快速开始(默认开图)](https://docs.mthreads.com/vllm-musa/vllm-musa-doc-online/S4000/%E5%8E%86%E5%8F%B2%E7%89%88%E6%9C%AC%E4%BF%A1%E6%81%AF/quick_start) · [vllm-musa PyPI](https://pypi.org/project/vllm-musa) · [S4000 release notes](https://docs.mthreads.com/vllm-musa/vllm-musa-doc-online/S4000/releasenote) · [vllm-omni MUSA 平台](https://docs.vllm.ai/projects/vllm-omni/en/v0.20.0/api/vllm_omni/platforms/musa)
- 教程/解析:[aiengineeringfromscratch CUDA graphs](https://aiengineeringfromscratch.com) · [vLLM 调度深度解析](https://audreywongkg.medium.com/understanding-vllm-scheduling-token-budgets-chunked-prefill-and-policies-2c879e3980e3) · [chunked prefill 解析](https://www.spheron.network/blog/chunked-prefill-explained-how-it-smooths-ttft-under-load-202) · [Sarathi 笔记](https://github.com/junuxyz/mlsys-notes/blob/main/notes/sarathi-serve.md)
