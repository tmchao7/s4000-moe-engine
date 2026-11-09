# P30-A Day2：8k 长上下文质量门（30B）

> 日期：2026-11-09。范围：P30-A 质量门/API 校验；SLO 并发探针因交接中止，未出结果。
> 原始 JSON：`benchmarks/data/P30/longctx_quality.json`（`benchmarks/data/` 不入库）。
> 脚本：`scripts/p30_longctx_quality.py`、`scripts/p30_longctx_slo.py`。

## 1. 预注册门

- 上下文约 7.6k tokens；needle 放在 10%/50%/90% 深度；3 个唯一 code；
- 贪心生成 12 token；**9/9 输出包含正确 code 才算 PASS**；
- 生产栈 TP2/EP2、`grouped_mm` prefill、`bmm_dense` decode、`triton_paged`、eager、chunk 256。

## 2. 结果：9/9 PASS

| 指标 | 值 |
|---|---:|
| prompt tokens | 7615 |
| cases | 9（3 codes × 3 depths） |
| passed | **9/9** |
| 整批耗时 | 42.207 s |

示例输出：

- code=4821 depth=0.9 → `4821\n\nThe secret code is 48`；
- code=7395 depth=0.1/0.5/0.9 → 均以 `7395` 开头；
- code=1568 depth=0.1/0.5/0.9 → 均正确。

说明：这是**合成 needle-QA 质量门**，证明 8k 上下文里模型能取到关键信息；不等于完整长文能力评测（多跳/摘要/引用仍需真实数据）。

## 3. API 超长输入校验

用 0.6B、`max_model_len=2048` 的服务实测：

```json
{"prompt": [1] * 2048, "max_tokens": 8}
```

返回：

```text
HTTP 400
{"detail":"prompt(2048) + max_tokens(8) exceeds max_model_len(2048)"}
```

说明 API 的超限拒绝路径是通的；没有静默截断/排队。

## 4. SLO 并发探针（未完成）

- 已写 `scripts/p30_longctx_slo.py`：18 条独立 7.6k prompt，记录 TTFT/e2e/aggregate；
- 本轮启动后因交接需要**主动中止**（30B 正在加载，尚未产生结果），**没有可引用的 SLO 数字**；
- 新 Agent 应重跑：

```bash
ssh s4000 'cd /root/moe-server-s4000 && PYTHONPATH=. \
  python3 scripts/p30_longctx_slo.py --output /root/p30_data/longctx_slo.json'
```

## 5. 结论

1. 8k 合成质量门通过 9/9；API 超限 400 通过。
2. 8k 仍不是默认：Day8 单条 prefill 约 14s，交互 SLO 需真实并发数据；SLO 未完成前不能宣称产品可用。
3. 生产默认未变，服务器已清场（双卡 0 MiB/无进程）。