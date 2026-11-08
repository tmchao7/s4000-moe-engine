# P22-A — 采样面补齐(对齐 vLLM/SGLang 服务层)

日期:2026-11-08 | 门:PLAN.md P22-A(预注册) | 测试:`tests/p22_sampler_face.py`(t0–t7)、`tests/p22_baseline_greedy.py`(改造前基线)、`scripts/p22_api_smoke.py`(API 冒烟)、`scripts/archive/p22_30b_final.py`(30B 收官窗口) | commit `89536b8`(引擎)/`4fd27fe`(api)/本报告

## 0. 一句话

采样面从"只有 temperature"补齐到 vLLM/OpenAI 对齐:**top_p / top_k / min_p / presence·frequency·repetition penalty / stop_token_ids(引擎级) / logprobs(top-k) / skip_special_tokens / `/v1/completions`(流式+非流式)**;全部纯 torch 算子(MUSA 零厂商风险),默认参数走逐字 legacy 路径(bitwise 无操作门实测通过),所有新能力 per-request opt-in,生产栈一个默认没动。

## 1. 背景(差距盘点)

本轮前置盘点(调度/服务层 vs vLLM/SGLang,证据 file:line):调度侧(stall_free/chunked prefill/优先级/LRU/混合批/PD)与 stop 匹配位置(API 层事后匹配 = vLLM V1 同位)均已对齐;最大服务面缺口是采样面——`sampler.py` 全文 25 行仅支持 temperature,`sampling_params.py` 4 字段,`/v1/completions` 不存在,logprobs/tools 显式 400。盘点同时确认 **overlap scheduler(vLLM V1)与 xgrammar 结构化输出**是下两个对齐目标(本轮用户裁决只做采样面,二者入 backlog)。

## 2. 实现(按层)

| 层 | 内容 |
|---|---|
| `sampling_params.py` | 新字段 9 个,`__post_init__` 校验(top_p∈(0,1]、top_k=-1 或 ≥1、min_p∈[0,1]、penalty 范围、logprobs≤20、stop_token_ids 整数表),非法值 ValueError → API 400 |
| `layers/sampler.py` | 流水线 **penalty → temperature → top_k → top_p → min_p → exponential-trick 采样**(vLLM 同序);`SampleMeta` 承载每步旋钮(host 侧构建,rank0 only);掩码用 `finfo.min` 不用 -inf(MUSA softmax 有限性);top_k/top_p 共享一次 sort,per-row 旋钮向量化;**is_neutral ⇒ 逐字 legacy 路径**;sampler 纯净化(fp32 输入不再被原地变异——`float()` 对 fp32 返回原张量的别名陷阱) |
| `engine/sequence.py` | 旋钮存储;`gen_counts`(生成 token 计数,append_token O(1) 增量)+ `rep_counts`(prompt+生成,懒构建——PD 交接后补 num_prompt_tokens 仍正确) |
| `engine/scheduler.py` | postprocess 增加 `stop_token_ids` 检查(**不受 ignore_eos 抑制**,同 vLLM;仅模型 eos 受其门控);事件元组 4→5 元(带 logprobs) |
| `engine/model_runner.py` | `_build_sample_meta`(penalty 张量对:repetition 见 prompt+生成 vLLM/HF 语义、presence/frequency 仅生成 token OpenAI 语义,同对集合并时按 is_gen 分配系数);`_sample_to_host`(采样+单次 D2H 同步点不变,logprob 行按请求过滤);`run()` 返回 `(token_ids, logprobs)`;`_run_mixed` 每子组重建 meta(行序=seqs 序已核) |
| `engine/llm_engine.py` / `async_engine.py` | step 解包 + TokenEvent.logprobs |
| `entrypoints/openai/api_server.py` | `_parse_sampling_kwargs` 共享解析(chat: bool logprobs+top_logprobs;completions: int logprobs);`POST /v1/completions`(str/token-id prompt、SSE `text_completion.chunk`、include_usage、stop 匹配同 chat);chat logprobs 输出 OpenAI content 形态(token/bytes/top_logprobs),completions 输出 legacy 数组形态(tokens/token_logprobs/top_logprobs/text_offset);tools/tool_choice/echo/suffix/best_of 维持 400 |

## 3. 门与判决(预注册,全部照实)

| # | 门 | 结果 | 判决 |
|---|---|---|---|
| 1 | t1/t1b/t3:掩码/penalty 对独立 CPU 参考(支撑集相等+allclose+逐位) | 10 掩码用例 + penalty 用例全过 | PASS |
| 2 | t2:支撑集封闭性(4 旋钮 × 200 抽样,0 泄漏) | 首跑 FAIL→根因是**测试漏了温度先行**(top_p 核随 T=1.5 展宽),修测试后过 | PASS |
| 3 | t4:全默认 meta(None 与 neutral 两种)与逐字 legacy 体在相同 RNG 种子下逐位相等 | 3 种 all_greedy 模式全等 | PASS |
| 4 | t5:0.6B 默认 greedy 与改造前捕获基线逐位全等(改造前 `p22_baseline_greedy.py capture`,改造后 compare) | 8 prompt × 48 token 全等 | PASS(无操作硬门) |
| 5 | t7:默认参数 decode 步时 vs 改造前基线 | 52.46ms vs 52.21ms = **+0.47%**(≤±1% 门) | PASS |
| 6 | t7:top_p=0.9 激活 TPOT 开销(交错 ×3 ABA) | ratio **1.091**(≤1.10 门;全词表 sort 成本,0.6B 词表 151936) | PASS(贴门,照录) |
| 7 | API 冒烟:旋钮 200/logprobs/400 校验/completions+SSE/stop_token_ids | 6/6 OK(SSE 17 chunks;logprobs 形态含 bytes) | PASS |
| 8 | 回归 | v3 28 passed / p8 / p2 / p6 / p16-mixed / p16-lru / p16-priority(CPU t1–t6;**引擎门=P16-C 已照录负结果,`--skip-engine` 口径**)/ p17-unit | 8/9 PASS |
| 9 | 30B 收官窗口(单载) | 见 §4 | 见 §4 |

**回归期间抓到并修复**:① `_run_mixed` 未防 logprobs=None(p16-mixed 回归逮住,e2e 不走 mixed 路径——回归网的价值实证);② detokenizer 类参数化漏改 `__init__`(API 冒烟逮住);③ `top_ks` int32 进 gather 需转 int64(自查)。

## 4. 30B 收官窗口(单载,生产栈:TP2/triton_paged/bmm_dense/stall_free/eager)

一次 30B 加载窗口(`scripts/archive/p22_30b_final.py`,数据 `/root/p22_data/final_30b.json`),**全程 PASS**:

| 检查 | 结果 |
|---|---|
| 默认 greedy 自一致(同载两跑,I34 口径,间清前缀缓存) | **逐位全等 True** |
| decode 步时(16 seqs × 64 tok × 3 段) | 段中位 181.48/181.57/181.67 ms,**中位 181.57 ms,CV=0.0004**(≤5% 门) |
| top_p=0.9 sanity | True(2 prompt × 32 token 完成) |
| 引擎级 stop_token_ids | True(确定性 2-token 停止,finish="stop") |
| logprobs sanity | True(n=8,greedy top1 id == 采样 id) |

30B 采样链路(含新流水线与事件通路)在生产栈上行为正确;步时绝对值 181.57 ms@b16 为本轮新口径记录点(b1 142 ms 系历史单批口径,不可比)。

## 5. 性能注记

- 默认路径:**逐字 legacy 代码**(meta=None 或 is_neutral 直接走原表达),零分支开销;实测 +0.47%(过程噪声内)。
- 旋钮激活的代价集中在:top_k/top_p 一次全词表 sort(0.6B 151936 词 × 行数)、penalty 的 pair 张量 H2D(每步 host 重建,O(prompt+生成 unique));penalty 常开+长 prompt 场景的 host 成本与 vLLM 的 GPU 常驻缓冲方案有差距,记录为已知取舍(批量惩罚+超长 prompt 时优先考虑把 pair 缓存到设备侧)。
- logprobs 的 topk/log_softmax 仅在被请求时进入。

## 6. backlog(本轮盘点确认,未排期)

| 项 | 说明 |
|---|---|
| overlap scheduler(vLLM V1/SGLang 同款) | schedule+prepare 与 GPU forward 重叠,串行临界路径只剩 tolist+postprocess;旗舰件但动 step 主循环(I35 老地方),需 opt-in+bitwise 门+抢占回退同步设计 |
| 结构化输出(xgrammar) | vLLM/SGLang 同款后端;语法编译在 CPU(wheel 可装),逐步 bitmask 用 torch 算子上 MUSA(shift/and/where/masked_fill);per-request opt-in |
| n>1 并行采样 | fork+块引用计数;前缀缓存部分覆盖,共同到达时仍省重复 prefill |
| seed/per-request RNG | MUSA RNG 语义属厂商风险,缓 |
| tools/tool calling | 模板+解析层 |
| swap 型抢占 / KV offload | 现为 recompute 型;容量充裕时收益有限 |
| PD 流式输出/多实例 | P18-C 剩余 |
