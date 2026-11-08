# P24 — 结构化输出(xgrammar,vLLM/SGLang 同款后端)

日期:2026-11-08 | 门:PLAN.md §0a P24(预注册) | 测试:`tests/p24_guided.py`、`scripts/p22_api_smoke.py`(7/8 号用例)、`scripts/archive/p24_30b_final.py` | commit 本轮

## 0. 一句话

**结构化输出落地(per-request opt-in)**:`guided_json`/`guided_regex`/`response_format: json_schema`;xgrammar 0.2.7 编译(CPU,按 schema 缓存)+ 逐步 bitmask 用纯 torch 算子上 MUSA(LSB-first 展开 + where 掩码);掩码语义与 xgrammar 官方 CPU apply **支撑集逐位一致**;四 schema × 4 样本 100% 合法 + grammar 终止,约束开销 **1.028×**(≤1.10 门),不约束路径 bitwise 不变。

## 1. 实现

| 层 | 内容 |
|---|---|
| `sampling_params.py` | `guided_json`(dict/str)、`guided_regex`(互斥校验);仅数据,引擎侧编译 |
| `engine/llm_engine.py` | `_attach_grammar`:懒加载 xgrammar + `TokenizerInfo.from_huggingface` + `GrammarCompiler`;编译缓存 `(kind, 规范化串)`;`GrammarMatcher` 挂 Sequence(driver-side,不过 TP pickle);bitmask 词数 stash 到 runner |
| `engine/model_runner.py` | `_build_sample_meta`:`fill_next_token_bitmask`(pinned int32 [n,W],同一次 H2D)→ `SampleMeta.grammar=(rows, bitmask)` |
| `layers/sampler.py` | `_apply_grammar`:温度除法**之后**、top_k 之前(finfo.min 经再除会溢出 -inf);位 t = 字 t//32 的第 t%32 位(LSB-first);tokenizer 词表(151669)< logits 宽(151936)的尾部 padding 全掩;greedy 路径同样作用(掩码改变 argmax,与 top_* 不同) |
| `engine/scheduler.py` | postprocess:`accept_token(token)` 推进状态机,`is_terminated()` → finish "stop"(JSON 完成即停;eos 在对象完成后进入支撑集,自然 eos 规则兜底) |
| `entrypoints/openai/api_server.py` | `response_format:{type:"json_schema", json_schema:{schema}}`(兼容裸 schema 对象)+ `guided_json`/`guided_regex` 透传;非法 → 400;xgrammar 缺失 → ValueError → 400 |

## 2. 门与判决

| # | 门 | 结果 | 判决 |
|---|---|---|---|
| 1 | g1 位掩码语义:torch 展开 vs xgrammar 官方 CPU `apply_token_bitmask_inplace` | 支撑集逐位相等(7 token)+ accept/reject 一致(新 matcher 逐例) | PASS |
| 2 | g2 JSON schema:4 schema(含嵌套 array/enum/number)× 4 样本 | 16/16 `json.loads`+`jsonschema` 合法,全部 grammar 终止(finish=stop) | PASS |
| 3 | g3 regex `[A-Z]{3}-[0-9]{3}` × 2 | 100% fullmatch + 提前终止(<48 token) | PASS |
| 4 | g4 不约束路径不变 | SampleMeta 无 grammar 即中性路径(p22 t4/t5 门覆盖)+ 回归全绿 | PASS |
| 5 | g5 约束开销:8 seqs 约束 vs 不约束 decode 交错×3 | **1.028×** ≤ 1.10 | PASS |
| 6 | g6 参数校验 | 非法类型/互斥 → ValueError(→400) | PASS |
| 7 | API 冒烟 7/8 号用例:response_format 经 HTTP 端到端 | 合法 JSON 返回 + 非法类型 400 | PASS |
| 8 | 30B 单载 guided sanity | 见 §4 窗口 | PASS |

**测试构造注记(照录)**:JSON 字符串值内部语法放行全部 token,temp 0.8 的 0.6B 会把不限长的字符串写穿 max_tokens——测试 schema 加 `maxLength/maxItems` 约束(有界性本身是 schema 的一部分,非弱化门);离线 `generate()` 的 `tokenizer.decode` 在 transformers 5.13 上不跳特殊 token(Qwen 词表怪癖),API 增量解码路径(skip=True 默认)不受影响。

## 3. 已知边界

- `json_object`(任意合法 JSON)未接(builtin grammar 编译入口待接,backlog);`guided_choice`/structural tags 未接。
- 每步 bitmask 展开为 [n, W, 32] torch 算子(W=4740),约束行多时开销线性(g5 的 1.028× 为 8 行口径);xgrammar 有 CUDA/Triton 后端,MUSA 无,如成瓶颈可写 triton-mtgpu 内核(backlog)。
- 语法编译在 API 进程 CPU,首次 ~10ms 级/请求(schema 复杂度相关),缓存后为 0。
