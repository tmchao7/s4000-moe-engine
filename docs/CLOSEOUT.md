# CLOSEOUT — 遗留问题与状态总账(2026-11-08,P21 后)

> 本文只做一件事:把"还剩什么"一次说清。技术细节以 [ISSUES.md](./ISSUES.md) 为准,当前状态以 [HANDOFF.md](./HANDOFF.md) 为准,厂商沟通物料以 [E-vendor-ticket.md](./E-vendor-ticket.md) 为准。

## A. 已解决(累计亮点,每轮报告见 benchmarks/)

| 项 | 结果 |
|---|---|
| 生产栈定型 | triton_paged + grouped_mm/bmm_dense + eager + stall_free + TP2/EP2;30B 双卡 100% SLO,在线 ≥1.07 req/s,离线 141.5 tok/s(b64) |
| 调度/服务化 | stall_free + chunked prefill + 前缀缓存 + LRU 淘汰 + 优先级/SLO 分级(高优 TTFT 42.7×)+ host-sync 移除 + 统一混合批(混合步 −25%/−15%,opt-in) |
| 采样/服务面(P22) | top_p/top_k/min_p + 三种 penalty + 引擎级 stop_token_ids + logprobs + /v1/completions(流式);默认参数逐字 legacy 路径(改造前基线 bitwise 全等),全 per-request opt-in |
| PD 分离 | 单机双卡 1P1D 全链路:bitwise 交接 + 在线 A/B ITL p95 0.753×/0.839×;Poisson 双档 POSITIVE(0.79×/0.82×);P 侧前缀复用 5.7×;OpenAI 端点 |
| 平台适配 | MUSA 全栈移植 + Triton-mtgpu 内核(paged/fused-moE)+ I28/I31/I34 等平台坑闭环 |

## B. 无法解决 — 厂商问题(我们修不了,需要摩尔线程)

| # | 条目 | 为什么我们修不了 | 当前绕过 |
|---|---|---|---|
| 1 | **I29/E4 引擎级 MUSAGraph+MCCL replay 间歇楔死** | 闭源 driver/runtime/MCCL 层;P14 证实 MCCL 混用是成分之一(缓解 0→33%)但非全部;无诊断工具可见 | `enforce_eager=True`(生产默认);分段图仅单流可显式开 |
| 2 | **fp8 e4m3 cast 在 MUSA 静默 no-op**(P20 新) | torch dtype 存在但 cast 不量化(492 roundtrip 零误差=e4m3 max 448 必 clip,故为假实现)——静默错误比崩溃危险 | **不使用 fp8**;int8 KV 的替代方案见 I20 |
| 3 | 图可执行上限 ~2048/进程,torch_musa 直连无轮换 | 驱动限制;torchada 才有 LRU 轮换 | 分段图 dense 桶 1056 图/rank 在上限内 |
| 4 | E3 `torch._grouped_mm` 无旋钮、不可捕获 | 厂商 op | decode 用 bmm_dense、prefill 保持 grouped_mm |
| 5 | E5 mp22 fast path 数值错误(rel 1.3–341、b64 NaN) | 厂商 kernel | `use_mp22=False` |
| 6 | E6 launch watchdog 10 min、报错点漂移 | 驱动参数不可配 | 自研硬 watchdog + P14 exit 看门狗 |
| 7 | I28 torch.musa 跨块 strided view D2H 返回垃圾 | 平台 bug 嫌疑 | `.contiguous()`/index_select 连续化(已固化进 kv_channel) |

厂商沟通物料:[E-vendor-ticket.md](./E-vendor-ticket.md)(I29 楔死分级探针 + 2048 交叉印证 + MCCL 混用实测 + fp8 no-op 最小复现)。**外发必须由用户亲自操作。**

## C. 已判决放弃(不要重开,证据在 benchmarks/)

| 项 | 终判 |
|---|---|
| **I20 int8 KV(三种方案)** | per-token-head / per-channel 校准 / 混合精度 outlier 通道——机制全部逐位验证正确,质量门连续 FAIL:**Qwen3 K/V 通道内动态范围 ~500× 超出 int8 的 127 级绝对网格**;除非厂商 fp8 落地,否则放弃容量 2× 奖池 |
| 投机解码 / EAGLE3 / S 系列 | α≈0 且 c=0.39,NO-GO |
| I19"launch-bound→融合换步时" | 实测否定(融合 −0.76%) |
| 分段图(整图替代) | 离线 −23.7% 但在线 FAIL(padding 税+AR 串行化),定格单流开关 |
| E1 kernel autotune | 内核 2× 但翻转 0.81%>0.5% 冻结门;30B e2e 复核 1.01×(负)→ tuned 表维持 opt-in |
| I30 / I33 / profiled placement | 回归主张否定 / padding 税+AR 串行化 / 不采用 |
| 未查但低收益 | I31(30B 首 batch 跃迁,一致性门丢弃首 batch 兜底) |

## D. 需要用户亲自做

1. **审本地 commit 并 push**:`git log --oneline origin/main..HEAD` 一览(P19–P21 共 8 个未审)。
2. **发厂商工单**:[E-vendor-ticket.md](./E-vendor-ticket.md)(含 P20 fp8 no-op 新证据)。
3. **裁决两件产品决策**(非技术问题):
   - P16-A 统一混合批**默认翻转**提案:8B/30B 逐位 0 翻转 + 性能双 PASS,唯一瑕疵是 0.6B 翻转 0.56%>0.5% 冻结门——需你接受"0.6B 小模型翻转"或维持 opt-in。
   - 旧 seetacloud 实例释放(宿主级 22GB 无主显存只有平台能清)。
