# 开发与验证

> 这一页面向维护者，不是部署必读。产品安装/启动见 [DEPLOY.md](DEPLOY.md)，运维见 [OPERATIONS.md](OPERATIONS.md)。

## 1. 开发环境

```bash
pip install -e ".[test,plot,serve]"
python3 -c "import torch, torch_musa; print(torch.musa.is_available())"
```

`test` extra 含 pytest/xgrammar/jsonschema/httpx；`plot` 只用于历史图表重渲；`serve` 是 API 运行依赖。

## 2. 常跑正确性门

```bash
PYTHONPATH=. python3 tests/p2_grouped_mm.py
PYTHONPATH=. python3 tests/p6_bmm_dense_tiny.py
python3 -m pytest tests/test_benchmarking_v3.py -q
PYTHONPATH=. python3 tests/p8_scheduler_gaps.py
PYTHONPATH=. python3 tests/p25_overlap_prefill_boundary.py
```

## 3. 全量回归

```bash
# 本机 MUSA 环境（16 套）
bash tests/run_regression.sh --model /models/Qwen3-0.6B

# 从本地同步到 s4000 再跑（开发用；需要 ssh alias s4000）
bash scripts/remote_regression.sh
```

回归包含 scheduling、overlap、paged attention、MoE、采样面、结构化输出、API 等；不是部署前置。

## 4. 性能/对比实验

历史实验报告在 [benchmarks/](../benchmarks/) 与 [BENCHMARKS.md](BENCHMARKS.md)；冻结门/判决见 [HANDOFF.md](HANDOFF.md)、[PLAN.md](PLAN.md)、[ISSUES.md](ISSUES.md)、[CLOSEOUT.md](CLOSEOUT.md)。

```bash
PYTHONPATH=. python3 scripts/p2_benchmark.py --help
PYTHONPATH=. python3 scripts/benchmark_online.py --help
python3 scripts/render_metrics.py
```

`benchmarks/data/` 不入库；新 clone 重渲图表需要先自产 measurements。

## 5. 文档与发布检查

```bash
python3 scripts/check_docs.py
python3 -m py_compile $(find nanovllm -name '*.py') tests/*.py scripts/*.py
bash scripts/build_release.sh
```

发布包构建要求工作树干净（从 HEAD 打包），并断言不包含 `benchmarks/`、`tests/`、`assets/`、`scripts/archive/`。

## 6. 设计/平台资料

- [architecture.md](architecture.md)
- [numerics-policy.md](numerics-policy.md)
- [musa-pitfalls.md](musa-pitfalls.md)
- [TEST-MATRIX.md](TEST-MATRIX.md)
- [P25-review.md](P25-review.md)
- [engine-vs-vllm-sglang.md](engine-vs-vllm-sglang.md)（历史对标，非产品承诺）
- [ROADMAP-NEXT.md](ROADMAP-NEXT.md)（P29 评估 + 外部引擎调研 + 9 天候选任务）
- [P30-OPPORTUNITIES.md](P30-OPPORTUNITIES.md)（P29 后的机会评估：部署硬化 / 8k 长上下文）