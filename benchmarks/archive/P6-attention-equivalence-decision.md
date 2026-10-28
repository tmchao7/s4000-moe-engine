# attention 后端等价性决策（三层门）

| 层 | 检查 | 值 | 阈值 | 结果 |
|---|---|---:|---:|---|
| L3 | first_token_agreement | `1` | 0.99 | ✅ |
| L3 | degenerate_outputs | `0` | 0 | ✅ |
| L3 | qualitative_equivalence_ratio | `1` | 0.9 | ✅ |
| L3 | review_degraded_count | `0` | 0 | ✅ |
| L4 | per_backend_determinism | `{'sdpa': True, 'triton_paged': True}` | both true | ✅ |
| L2 | top1_flip_rate | `0.00212814` | 0.005 | ✅ |
| L2 | flip_margin_max | `0.3125` | 0.5 | ✅ |

只报告不作门：{"request_token_identical": 20, "request_token_agreement_mean": 0.509296875, "edit_distance_ratio_p50": 0.23316420300149962, "edit_distance_ratio_max": 0.8526315789473684}

**决策：adopt_triton_paged**
