# P11 — 在线侧填充项(I30 v4 口径复核 + Poisson 饱和 bracketed)

> 判定规则**先于本轮既定**(ISSUES I30 决策树 + P9-serving 饱和缺口),非跑后定制。背景:v4-era(P8 改动前,commit `d5779b6`)引擎整树部署于服务器 `/root/moe-server-p6d`,对它重放 v4 同矩阵(`--mode poisson --max-qps 0.87 --fractions 0.5,0.75,0.9,1.0 --requests 128 --prompt-words 128 --repeats 3 --request-timeout-s 120`,seed 20261026 = v4 同款),客户端 = 现行 benchmark_online(v3 schema,纯测量侧)。

## 判定规则(既定)

**I30 v4 口径复核**(1.0 档 @0.87 qps 的 TPOT p95;v4 单跑 = 0.327,现行栈 ×3 = 0.387–0.397):
- 旧栈 ×3 落 **≈0.33**(与 v4 单跑相容)→ **回退真实存在**,窗口 = P8 期累积改动;下轮做 P8 二分。
- 旧栈 ×3 落 **≈0.39**(与现行栈重叠)→ **v4 单跑口径假象,无回退,I30 关闭**。
- 中间地带(0.35–0.38)→ 部分回退,记录并缩窗口。

**Poisson 饱和 bracketed**(现行默认栈,`--fractions 1.2,1.4 --repeats 3`,I13 fraction>1 = max-qps 乘数 → 1.044 / 1.218 qps):
- 报告 slo_standard 达 99% 的最大达成 QPS 与 SLO 曲线;P9 四档全 100% 故饱和点未定,本轮补上界。

## 实际

**I30 v4 口径复核**:**中止,判定未产出**(紧急收尾)。
- 已完成:pre-P8 树(`d5779b6`)整树部署 `/root/moe-server-p6d`、旧栈服务健康(weights 338–348 s 加载)、同矩阵客户端启动。
- 实况:**旧栈 0.5 档单跑 ~12 min(vs 目标 ~5 min)**——比现行栈慢得多(与 I24 host-sync 修复在该树中缺席一致);收尾窗口内仅完成 0.5 档 ×2(128/128、SLO 100% ×2,无 TPOT p95,客户端 JSON 未写出)→ 按紧急收尾中止、按 PID kill、旧栈目录已删。
- 结论:无判定数据;**转下轮**(预算须按旧栈实速估:全矩阵 ×3 ≈ 2h+;或只跑 0.9/1.0 档 ×3 ≈ 50 min)。判定树不变(见上)。
- 证据:`benchmarks/data/p11/p11_i30_p6d_client.log`(部分)。

**Poisson 饱和 bracketed**:**未跑**(时间盒让位于 I30 复核)。转下轮,与在线 A/B 同服务顺路跑(`--fractions 1.2,1.4 --repeats 3`)。
