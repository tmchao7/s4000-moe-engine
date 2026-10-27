# 发给接手的 agent（复制下面整块）

```
你接手一个 MUSA（摩尔线程 S4000 双卡）上的 30B MoE 推理引擎项目。

仓库：/Users/tmchao7/repos/02_personal/moe-server-s4000（分支 feat/musa-port，HEAD 42f848a，工作区干净）
服务器：ssh -p 28036 root@connect.sha1.seetacloud.com（也可 `ssh s4000b`），代码在 /root/moe-server-s4000
远端 .git 是伪仓库，绝不可在其上提交；以本地为准，用 scp 同步。

===== 第一步：先读，别先动手（15 分钟）=====
1. docs/HANDOFF-P6d.md        ← 唯一权威交接，先读全文
2. docs/PLAN.md               ← 唯一计划（Track A/B/B'/C/D + 冻结门汇总 + 裁剪决策树）
3. docs/ISSUES.md             ← 20 项台账（I/S/E 三系列）
4. 需要时：docs/P6-numerics-policy.md（I4 数值契约）、benchmarks/P7-phase0.md（P7 已判死）
5. git status && git log --oneline -3   ← 确认 HEAD=42f848a，不要 reset
6. ssh -p 28036 ... 'mthreads-gmi | grep -E "^[01] "; pgrep -fa "^python3"'
   ← 期望双卡 0 MiB、无 python 进程；若有残留按 HANDOFF-P6d §4 清理

===== 当前状态（一句话版，读文档核对）=====
- 已落地：decode MoE 用 bmm_dense（离线 b32 110.6 / b64 141.5 tok/s），b32/b64 不稳定已由 shm 竞态修复治好。
- 已判死：投机解码 NO-GO（α=0.006、c=0.384、最优 0.73×，比基线还慢）；图路径控制组 0/5（禁 30B A/B）。
- 卡住：paged attention 已过 L1–L4 等价门（决策 adopt_triton_paged），但 L5（在线性能）没跑成
  —— 因为 `api_server --attention-backend triton_paged` 启动期挂死。默认仍是 sdpa，+35~59% 还挂在桌上。
- 新发现：I19 每步固定成本（0.6B 步时 90% 是层内固定开销；30B b1 的 175 ms 里 forward 占 95.5%）。

===== 任务（按序，共约 3.5 小时；时间盒到了就停并记录）=====

P0（~70 分钟，最高价值）：把 I4 的 L5 做完
  目标：把 adopt_triton_paged 落成默认，或明确否决。
  步骤 1（~30 min）排查 paged 启动挂死：
    - 现场：api_server + triton_paged 启动 15 分钟未起；两 rank 权重已载（各 29.5 GiB）、零日志输出、
      GPU 0% util、/health 无响应。同代码用 sdpa 启动正常。
    - 做法：在 nanovllm/engine/model_runner.py 的 __init__ 里，warmup_model 前后、
      allocate_kv_cache 前后各加一行 print，并带 MUSA_LAUNCH_BLOCKING=1 重跑，定位卡在哪一步。
    - 45 分钟规则：查不出来就记录现场、转 P1，不要死磕，也不要重试超过 2 次。
  步骤 2（~30 min）L5 实测（参数与 benchmarks/P6-serving-v3.md 完全一致）：
    服务端：python3 -m nanovllm.entrypoints.openai.api_server --model $MODEL \
              --scheduler-policy stall_free --decode-moe-backend bmm_dense --attention-backend triton_paged
    客户端：python3 scripts/benchmark_online.py --model qwen3-30b-a3b --tokenizer $MODEL \
              --max-qps 0.87 --requests 128 --max-tokens 64 --prompt-words 128 \
              --arrival-pattern poisson --slo-ttft-ms 2000 --slo-tpot-ms 500 \
              --fractions 0.5,0.75,1.0 --output benchmarks/data/online-v4-paged.json
  门（冻结，不可改）：100% 档 SLO 达标率 ≥ 100%（v3 基线）；且 TTFT/TPOT 不劣化。
  过门则：改 nanovllm/config.py 与 api_server.py 的默认 attention 后端 → 生成
    benchmarks/p2-reference-8tok-paged.json（paged 自身的冻结参考）→ 写 benchmarks/P6-serving-v4.md
    → 更新 ISSUES I4 为已关闭、E1 解除闸门。不过则：默认保持 sdpa，I4 关闭为"L5 否决"，如实记录。

P1（~20 分钟）：I19 每步固定成本的定向测量
  跑 0.6B（/root/autodl-tmp/models/Qwen3-0.6B）的逐层计时（28 层逐层前向），判定是
  "launch 次数多" 还是 "单个 kernel 慢"。
  判读：若每层耗时均匀且 ≈1.4–2.2 ms → launch-bound（图会变高优）；若个别层尖峰 → 定位具体 kernel。
  注意：一次只装一个模型（同进程连开两个会 no KV cache memory 并卡 barrier）；
        脚本必须有 if __name__ == "__main__"（否则 spawn 递归挂死）。

P2（~60 分钟，1 发 30B 加载）：I8 长上下文重测
  上一轮 I8 是无效跑：prompt_words=1024 被切成 18,354 token（不是意图的 ~1k），且 rank1 中途死亡，
  120–124/128 请求没拿到 [DONE]。
  重设计要点：用 token 数控制长度（目标 ~1k token）、显式 max_model_len、运行中监控 rank1 存活、
  失败请求单列分母。跑 3 档（0.435/0.653/0.870 QPS），写 benchmarks/P6-i8-long-context.md。

===== 硬约束（违反会浪费一整晚）=====
- 一次只跑一个 GPU 任务；30B 加载前：锚定 pgrep 看 PID → kill -9 → rm -f /dev/shm/nanovllm* → 双卡 0 MiB。
- 不要用 pkill -f / pgrep -f 的未锚定模式（会匹配到执行命令的 bash 自己，把 ssh 会话杀掉）。
  用 "^python3"、"^/bin/bash"、"[s]pawn_main"，或先 pgrep -fa 列 PID 再按 PID kill。
- 不要把 vllm_fused 设默认（慢 2.3–6.3×）；不要用 mp22（数值错）；
  不要全局设 moe_backend=bmm_dense（prefill 会 OOM，只能 decode_moe_backend）。
- 不要改 scripts/p5_decide.py 的任何阈值；不要改 site-packages。
- 不要 git push（本地 commit 即可）；不要 reset 工作区。
- 不要重跑已定案实验：投机解码（已 no-go）、30B graph A/B（控制组 0/5）、I4 的诊断（已定案）。
- 不要留自动重跑的等待器（上一轮有个 wait_p7.sh 在 GPU 空闲时自动重跑了整条链，浪费一次加载）。
- 每个实验先写"假设 → 唯一变量 → 预期 → 判定门"到日志，再跑；一次只改一个变量。

===== 自主迭代协议 =====
- 用户不一定在线，不要问问题：按证据选默认值，把理由写进日志。
- 单个问题卡住 >45 分钟：日志写清"已排除/未排除/下一步"，切下一条。
- 失败/负结果照样写进报告——"没做到"比"假装做到"有价值。
- 崩溃先救证据（JSONL/日志已在盘上），清理后再用同一命令重试一次；同一失败两次就换路径。
- 墙钟是约束：宁可少跑一次、把一次跑透，不要堆无判定门的实验。

===== 交付物 =====
1. 每个实验：JSON + 日志 + sha256，证据 scp 到本地 benchmarks/data/（该目录不进 git）。
2. 报告写进 benchmarks/（如 P6-serving-v4.md、P6-i8-long-context.md），结论写进 docs/。
3. 更新 docs/ISSUES.md（条目状态）、docs/PLAN.md（Track 进度），必要时新建 docs/HANDOFF-P6e.md。
4. 结束时本地 commit（不 push），message 写清"做了什么 / 证据在哪 / 哪个门没过"。
5. 结束时服务器必须：双卡 0 MiB、无残留进程、/dev/shm 干净。

===== 验收标准（用户会照这个看）=====
A. I4：paged 是否成为默认？如果没成，卡在哪一步、下一步是什么（必须有现场证据）。
B. I19：每步固定成本是 launch-bound 还是单 kernel 慢？给出数字（每层 ms）。
C. I8：长上下文（~1k token）的 TTFT/TPOT/SLO 数字，或明确说明为什么还是没测成。
D. 所有结论都有 JSON/日志 + sha256；服务器已清理；本地 commit 存在。
E. 不许出现"跑过了"但没有数字、"修好了"但没有证据的表述。

先读 HANDOFF-P6d.md，然后从 P0 开始。任何与文档冲突的地方，以代码/实测为准，并先修文档再继续。
```
