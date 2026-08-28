# MemGen V3：当前系统与执行流程

V3 系列共同冻结 layer 24、embedding key → compiled side-KV value、最多三次 retrieval attempt、
replace-current-memory、greedy decoding、bfloat16 与 SDPA。V2 及早期 V3 入口仍保留用于历史复现，
但当前主线是 V3.4 的逐 token entropy+risk 联合 gate，以及 V3.5 在其上增加的 applicability selector
与 terminal-abstain 生命周期修复。

| 版本 | gate / query | selector 与 abstain |
|---|---|---|
| V3.1--V3.3 | 语言学 boundary；question + full partial CoT | 全库动态检索与 margin 变体；历史实验 |
| V3.4 | 每个 pre-answer token；entropy 与 layer-24 risk 联合控制；current-token pooling | 全库动态检索；abstain 后已有 memory 继续 active |
| V3.5 | 完全复用 V3.4 gate 与 current-token full-prefix query | question-only applicability shortlist → shortlist 内 dynamic rerank；abstain terminal 且清除旧 memory |

## 1. 完整数据流

```text
离线阶段
MemoryRecord + Phase-1 verified source
  ├─ sanitized when_facing
  │    └─ 复用并逐条复现 V3 layer-24 / last-valid-token / L2 key
  │          └─ applicability key bank
  ├─ When facing + sanitized transferable_decision
  │    └─ frozen reasoner layer-24 / last-valid-token / L2
  │          └─ dynamic key bank
  └─ full When facing + Prefer + Avoid（保持原样）
       └─ frozen reasoner layer-24 canonical pre-RoPE
             └─ 已有 side-KV value bank；V3.5 不重编译

在线阶段
question.strip()（无 ChatML / prompt boilerplate）
  └─ 一次性 pure re-encode → applicability exact cosine → floor → frozen top-k shortlist

每个 pre-answer generated token t
  └─ entropy >= high 且 layer-24 persistence risk > threshold
       └─ 暂停 side-KV，重编码 prompt + 从首 token 到 t 的完整 partial CoT
            └─ 只在 frozen shortlist 的 dynamic keys 中 exact cosine top-2
                 ├─ margin 通过 → activate / replace / duplicate compiled side-KV
                 └─ margin 未通过 → terminal abstain；清除 active memory 并 native re-forward
```

embedding 只选择 `memory_id`；value 始终是该 ID 已编译的 side-KV。key 与 value 通过同一个
`memory_id`、`payload_hash` 和固定 record order 绑定。payload 与 provenance 保留用于审计，线上不用
BM25，也不把经验文本重新编译为 KV。

## 2. 离线阶段

原始 V3 key bank 精确编码 `MemoryRecord.sanitized_fields["when_facing"].strip()`；它不是
`sanitized_retrieval_key`，也不含 Prefer 或 Avoid。V3.5 把该 bank 作为 applicability embeddings 复用，
但必须验证 `key_source`、layer、pooling、normalization、逐 memory embedding hash 以及与 E0 side-KV 的
ID/order/payload 对齐，不能只相信已有路径名。

V3.5 另外通过 `source_experience_id` 一一 join Phase-1 approved/verified records，把
`bank.target.transferable_decision` 单独按 E0 规则清洗，再用 V3.5 的确定性
`canonicalize_answer_format_vocabulary_v1` 规则规范化其中的 boxed/final-answer 输出格式词汇，构造唯一允许的
dynamic key 文本。该局部规则不改变 E0 sanitizer，也不改写 `when_facing`；真实 GSM8K 记录里
`when_facing` 对“最终答案类型”的描述仍按原 V3 key 逐字复现：

```text
When facing: {sanitized when_facing}
Prefer: {V3.5-canonicalized sanitized transferable_decision}
```

`verification_rule`、reference、Avoid、generic failure signal 与 final-answer boilerplate 均不得进入 dynamic
key。compiler 在 layer 24 取 last-valid-token state 并 L2 normalize，同时 fail closed 验证 Phase-1、E0、
原 V3 key bank、reasoner/tokenizer revision 和 side-KV provenance。它输出 dual-key tensor/manifest、离线
报告以及只基于 source-question positive pair 的 applicability calibration。

source pair 按 `memory_id:source_experience_id`、seed `3501`、train fraction `0.8` 做确定性分区。train 上在
`k=1..min(32, memory_count)` 中冻结达到 own-memory Recall@k ≥ 0.95 的最小 k，并用 own-positive cosine
的 5th percentile 冻结 inclusive applicability floor；heldout 必须同时满足 Recall@k ≥ 0.95 与 positive
retention ≥ 0.90。这里的 floor 是 positive-retention floor，不是完整 relevance classifier；整个离线过程
不读取 task answer、reward 或 accuracy。

V3.4 token-risk artifact 是独立、只读输入：V3.5 原样复用其 high/low entropy、persistence/recovery
prototype、risk threshold 与 horizon 32，不重新训练或重新标定。

## 3. 在线阶段

### V3.4/V3.5 连续 token gate

gate 初态为 `ARMED`，并观察答案标记前的每个 generated token，不检查逗号、句号或换行：

- 仅 `ARMED && entropy >= high && persistence_risk > risk_threshold` 发起 retrieval attempt。
- selected 且无 active memory 时 activation；selected 到不同 ID 时 replacement；相同 ID 时 duplicate。
  duplicate 仍消耗 attempt，并保留当前 memory。
- 成功 selection 后进入 `DISARMED`；必须连续两个 `entropy <= low` token 才 re-arm。第二个 low token
  只完成 re-arm，不能在同一 token 触发。
- 每题最多三次 attempt；第三次后为 `EXHAUSTED`。answer marker 或 EOS 后为 `CLOSED`。
- static floor 后不足两个候选时，V3.5 从一开始就是 `EXHAUSTED`，不产生 attempt，并要求 native
  zero-attempt exact parity；不能把单候选 margin 定义为 infinity。

V3.5 的任何 dynamic admission failure 都是 terminal abstain：无论是否已有 active memory，立即进入
`EXHAUSTED`，之后不再 re-arm。已有 memory 时还必须 deactivate 并回滚 counterfactual probe，随后在
side-KV inactive 状态下用当前 token 做 native re-forward；因此 `t+1` logits/cache 与后续 attention trace
都不能残留旧 memory。V3.4 的历史语义不变：它的 margin abstain 会消耗 attempt，但保留已有 memory。

### Cache 与 query 隔离

- static query 只编码一次原始 `question.strip()`，不含 system prompt、ChatML、assistant prefix 或已生成
  token；side-KV 必须 inactive。
- dynamic query 是 canonical prompt token IDs 加从第一个 generated token 到当前 token 的全部 token IDs；
  禁止 last-96 截断。query encoder 从头重算 prefix，并用 `controller.suspend_memory()` 隔离 active memory。
- activation、replacement 和 terminal clear 都从处理当前 observation token 前的 native cache 分支，按审计
  合同 rollback/re-forward；side-KV 不写入 Hugging Face native cache。
- 任一时刻最多一个 active memory。V3.5 memory 持续到 replacement、terminal abstain、answer marker
  或 EOS。

## 4. 评估与日志

任务正式指标只有：

- 严格准确率：仓库官方 GSM8K first-boxed reward；
- 格式准确率：第一个 boxed answer 存在且可解析；
- 生成 token：逐题计数（包含首次 EOS）以及 total/mean/median/p95/p99/max 和配对差值。

`numeric correct but format invalid`、answer-marker suppression、late attempt、memory lifetime 与 token outlier
只作为描述性诊断，不能成为第三种正式准确率或在线 gate。

evaluator 每题 append、flush、`fsync`，resume 时认证 profile、artifact hashes 与逐行 hash。V3.5 日志除
completion 和任务指标外，还记录：一次性 static question hashes、pre/post-floor shortlist、每 token 联合 gate
状态、full-prefix dynamic query hashes、shortlist 内 top-2/margin、selected/duplicate/replacement/terminal
abstain、clear native re-forward 的 KL/top1/timing、memory transition/span/attention，以及所有 parity、budget、
re-arm、shortlist、KV 与 stale-attention 完整性计数。默认不保存 full logits 或 hidden states。

CPU analyzer 入口保持统一：

```bash
python scripts/analyze_v3_evaluation.py \
  --results "$RUN_DIR/results.jsonl" \
  --run-profile "$RUN_DIR/run_profile.json" \
  --output "$RUN_DIR/analysis_report.json" \
  --markdown-output "$RUN_DIR/analysis_report.md"
```

## 5. V3.5 压缩实验顺序与 final block

V3.5 runner 只执行下面四步，并对每一步的 schema、logical hash、provenance 与 qualification fail closed：

1. 一次 offline dual-key build/audit，冻结 shortlist k 与 applicability floor。
2. `calibration-val --limit 64` trace-only；static selector 已冻结，dynamic margin 尚不参与 admission，且
   calibration profile 明确禁止把任务结果用于 selector 决策。
3. 只用每题 first attempt 的 shortlist 内 margin，answer-blind 地冻结 50% retention dynamic threshold。
4. 用最终静态+动态 selector 与 terminal clear 跑完整 `dev-test --limit 0`（当前 473 题），然后 analyze、
   与兼容的 V3.4/V3.1 baseline 做 matched comparison，并生成 exploratory qualification。

V3.5 dev 已参与研究迭代，只能称为 `exploratory matched dev evaluation`。即使工程门槛全部通过，
qualification 也只能输出 `qualified_for_user_review=true`、`qualified_for_final_test=false`。runner 没有
`--run-final` 分支，绝不运行或复用 `final-test`；必须等用户看到 dev 报告后另行明确授权。

## 6. 最后处理的研究问题

### V3.1 selector-only 实验

V3 全量结果没有显示净收益后，下一步只修改 selector，不修改 layer-24、side-KV、entropy re-arm、
三次 attempt 上限、replacement 或 duplicate 语义。实验策略是用 calibration-val 首次 retrieval
attempt 的 top1-top2 margin 分布，answer-blind 地保留置信度最高的 50%；阈值不读取答案、reward、
strict 或 format 指标，并冻结到带 hash 的 calibration artifact。

先对现有 key bank 做一次 CPU-only 几何与 hubness 审计：

```bash
python scripts/audit_v3_retrieval_geometry.py \
  --retrieval-key-manifest "$V3_BANK_DIR/retrieval_key_manifest.json" \
  --memory-records "$E0_DIR/memory_records.v2.jsonl" \
  --output "$V31_DIR/retrieval_geometry_audit.json"
```

从完整、abstention-disabled 的 calibration-val 基线日志冻结阈值：

```bash
python scripts/calibrate_v3_margin_selector.py \
  --results "$CALIBRATION_BASELINE_DIR/results.jsonl" \
  --run-profile "$CALIBRATION_BASELINE_DIR/run_profile.json" \
  --retrieval-key-manifest "$V3_BANK_DIR/retrieval_key_manifest.json" \
  --target-retained-fraction 0.5 \
  --output "$V31_DIR/margin_selector_calibration.json"
```

低于阈值的 attempt 记录为 `abstained`，保留 top-2 审计信息但不加载或替换 KV；attempt 仍被消耗并
进入原有 `DISARMED/EXHAUSTED` 状态。使用同一 logical split 分别运行 disabled baseline 与 V3.1
后，做逐题比较：

```bash
python scripts/compare_v3_selector_evaluations.py \
  --baseline-results "$DEV_BASELINE_DIR/results.jsonl" \
  --baseline-profile "$DEV_BASELINE_DIR/run_profile.json" \
  --margin-results "$DEV_MARGIN_DIR/results.jsonl" \
  --margin-profile "$DEV_MARGIN_DIR/run_profile.json" \
  --output "$V31_DIR/dev_selector_comparison.json"
```

阈值必须来自 calibration-val；正式比较优先使用匹配的 dev-test baseline/V3.1。当前已经使用过的
official final-test 不得用于选择阈值，也不在这一轮重跑。

完整流程可由一个可恢复 runner 执行；已有完整 calibration/dev baseline 时可通过对应参数直接复用：

```bash
bash scripts/experiments/gsm8k/run_v3_1_selector_experiment.sh \
  --calibration-limit 0 \
  --dev-limit 0 \
  --target-retained-fraction 0.5 \
  --calibration-baseline-dir "$CALIBRATION_BASELINE_DIR" \
  --dev-baseline-dir "$DEV_BASELINE_DIR" \
  "$PHASE1_DIR" "$E0_DIR" "$RISK_ARTIFACT" "$OUTPUT_ROOT"
```

runner 只运行 calibration-val 与 dev-test，并分别生成 key audit、selector calibration、两条件分析及
matched comparison Markdown。

### V3.2 centered-retrieval 实验

V3.1 证明 margin abstention 可以过滤大量低置信检索，但 calibration-val 的 first-memory top-1
share 和 selection Gini 仍显示严重的在线 query→key hubness。V3.2 因此只改变检索表示空间：

```text
raw keys K                         raw online query q
    │                                    │
    ├─ centroid c = mean(K)              │
    │                                    │
    └─ normalize(K - c)                  └─ normalize(q - c)
                    \                      /
                     exact cosine top-2
```

centroid 只从已认证的原始 unit key bank 计算。keys 与在线 question + full partial-CoT query 使用同一
centroid；KV、MemoryRecord 和原始 key bank 均不重编译。layer-24、last-token pooling、entropy
re-arm、三次 attempt、replacement、memory score bias 和 side-KV 注入全部冻结。

centered calibration 必须重新运行 abstention-disabled calibration-val，因为 centered margin 与 V3.1
raw margin 不在同一数值空间。随后仍按首次 attempt、answer-blind、50% retention 冻结新阈值。正式
任务比较只运行 matched dev-test：

```bash
bash scripts/experiments/gsm8k/run_v3_2_centered_retrieval_experiment.sh \
  --calibration-limit 0 \
  --dev-limit 0 \
  --target-retained-fraction 0.5 \
  --v31-dev-dir "$V31_DEV_DIR" \
  --v31-selector-calibration "$V31_SELECTOR_ARTIFACT" \
  "$PHASE1_DIR" "$E0_DIR" "$RISK_ARTIFACT" "$OUTPUT_ROOT"
```

该 runner 复用完整 V3.1 raw-margin dev 结果，生成：

- raw 与 centered key-key geometry audit；
- centered calibration-val 无 abstention 轨迹；
- transform-bound centered margin calibration；
- answer-blind calibration stop gate：top-1 share 与 Gini 必须同时低于 V3.1，否则停止且不跑 dev；
- centered-margin dev-test 结果与完整分析；
- V3.2 minus V3.1 的 matched comparison，包括 calibration top-1 share/Gini、strict、format、token、
  mechanism 和 dominant memory payload。

首要判据是 centered calibration 的 top-1 share 与 Gini 是否同时下降。任务层面报告 paired strict
delta、bootstrap CI 和 McNemar p；不能只看点估计。V3.2 本轮不运行或重用 final-test 来选择方案。
如果 hubness 明显下降但 dev strict 没有改善，下一瓶颈转向 memory/KV 语义；如果 hubness 不下降，
下一步研究 query/key pooling 或文本构造，而不是 injection layer。

### V3.3 answer-blind pooling audit

V3.2 centering 将 calibration first-memory top-1 share 从 `0.4569` 降到 `0.3421`，但 selected memory
从 23 降到 12、Gini 从 `0.9573` 升到 `0.9760`，因此未通过 stop gate。V3.3 不继续使用 centering，
而是检查 last-token query 是否主要编码了触发符号 `,`、`.`、`\n`，从而造成 query/key 结构失配。

审计从完整、已认证的 V3.1 raw calibration 日志重建 418 个 first-attempt prefix。每个 query 只做一次
layer-24 forward，同时产生四个冻结候选：

| Candidate | Key pooling | Query pooling |
|---|---|---|
| `key_last__query_boundary_last` | 当前 key last token | 当前 boundary token；复现基线 |
| `key_last__query_pre_boundary` | 当前 key last token | boundary 前一个语义 token |
| `key_mean__query_partial_mean` | `when_facing` 全 token mean | partial CoT mean，排除 boundary |
| `key_mean__query_full_mean` | `when_facing` 全 token mean | full prefix mean，排除 boundary |

运行命令：

```bash
bash scripts/experiments/gsm8k/run_v3_3_pooling_audit.sh \
  --v31-calibration-dir "$V31_CALIBRATION_DIR" \
  --v31-selector-calibration "$V31_SELECTOR_ARTIFACT" \
  "$PHASE1_DIR" "$E0_DIR" "$OUTPUT_ROOT"
```

审计首先要求 prefix hash、原 key embedding hash、原 query embedding hash 和 raw top-1 memory ID 全量
复现，并要求聚合 concentration 与 V3.1 calibration artifact 完全一致。之后仅以 answer-blind geometry
准入候选：top-1 share 与 Gini 下降、selected memory 数不下降、normalized entropy 上升。候选按更低
Gini、更低 top-1 share、更多 selected memory 排序。

输出目录保存 summary Markdown、完整 JSON、逐样本 top-2 JSONL 及可复用的 key/query safetensors。
该 runner 不接受 logical split，不运行 dev-test 或 final-test；只有 audit 推荐候选后才另行实现在线
pooling、冻结 50% margin 并运行一次 matched dev-test。

### V3.3 pre-boundary 在线验证

pooling audit 的 161 个 key、418 个 query 和原 V3.1 top-1 均全量复现。唯一通过 geometry gate 的
`key_last__query_pre_boundary` 将 top-1 share 从 `0.4569` 降到 `0.2679`、selected memories 从
23 增到 27、normalized entropy 从 `0.4078` 增到 `0.4542`，因此进入一次 matched dev-test。

在线 profile 现在显式区分两种 query pooling：

- `last_valid_token`：V3.1 boundary-last 基线；旧 selector artifact 未记录 pooling 时只能解释为该模式；
- `last_token_before_trigger_boundary`：V3.3 候选。reasoner 仍重编码 question + full partial CoT 的完整
  token 序列，但在 layer 24 读取 boundary 前一个位置的 hidden state。

每次 retrieval attempt 同时保存 boundary token/id/text、被选作 query 的 token/id/text、完整 prefix
hash、query embedding hash、top-2、margin 和 memory transition。V3.3 selector artifact 同时绑定 raw
retrieval transform 与 pre-boundary pooling，因此 V3.1 margin threshold 无法误用于 V3.3。

为避免重复运行 1000 条 calibration generation，50% retention threshold 直接从已经通过认证的 418 条
`pooling_audit_samples.jsonl` pre-boundary margin 构建。builder 会重新验证 report/sample hash、候选资格、
逐样本 top-2、聚合 concentration 和 key-bank hash；任何不一致都会在 dev-test 前停止。

运行：

```bash
bash scripts/experiments/gsm8k/run_v3_3_pre_boundary_experiment.sh \
  --dev-limit 0 \
  --target-retained-fraction 0.5 \
  "$PHASE1_DIR" "$E0_DIR" "$RISK_ARTIFACT" "$OUTPUT_ROOT"
```

runner 优先复用完整的 V3.1 boundary-last margin dev 结果；若不存在，则用当前代码和原 V3.1 selector
补跑 matched baseline。随后只新增一次 V3.3 pre-boundary margin dev 运行，并生成两份完整 analysis 与
`dev_query_pooling_comparison.md`。比较报告包含 paired strict/format、bootstrap CI、McNemar、token delta、
activation/replacement/duplicate/abstain/re-arm、first-attempt concentration/margin 以及 boundary-token
strata；其中句号 `.` stratum 必须单独检查。runner 不自动运行 final-test。

### V3.4 continuous token entropy-risk gate

V3.3 将 first-attempt top-1 share 从 `0.4673` 降到 `0.2243`，但 matched dev strict 仍为
`-0.2114%`，说明仅移动 boundary 内的 pooling 位置没有带来任务收益。V3.4 因此不再把语言学
boundary 当作 gate 的候选集合，而是直接检验 entropy 与 layer-24 hidden state 能否在逐 token
尺度识别 persistence。

V3.4 使用独立 schema，不能把 V3.1--V3.3 的 boundary risk artifact 或 selector threshold 当作兼容
输入。冻结语义如下：

- 对每个答案标记前的已生成 token `t` 观测 final-layer sink-masked attention entropy `A_t`，并用
  layer-24 hidden state 计算
  `R_t = cos(h_t, persistence) - cos(h_t, recovery)`；不检查逗号、句号或换行。
- 仅当 `ARMED && A_t >= high && R_t > risk_threshold` 时发起 retrieval attempt。当前 token 不回滚；
  检索或替换的 memory 从 `t+1` 开始影响生成。
- attempt 后进入 `DISARMED`；必须连续两个 token 满足 `A <= low` 才 re-arm。第二个低熵 token 只完成
  re-arm，禁止在同一个 token 立即触发。
- attempt 上限仍为 3；activation 后的新 memory 替换旧 memory；duplicate 和 margin abstention 均消耗
  attempt 并保留当前 memory。
- query 仍是 question + 从首个生成 token 到 `t` 的 full partial CoT；layer-24 pooling 固定为
  `current_generated_token`。retrieval key bank、compiled side-KV bank、注入 layer 和 memory strength
  均不改变。

离线 compiler 在完整 experience 级别做固定 train/holdout 切分。high/low entropy threshold 只用
bank-train 的全部 pre-answer token；recovery horizon `H` 从 bank-train 高熵 burst 到“连续两个低熵
token”的恢复距离 p75 冻结（最大 32）。高熵 token 在 `H` 内出现稳定低熵段标为 recovery；完整观察
`H` 后仍未恢复标为 persistence；尾部不能完成判断时 right-censor。layer-24 prototype 只由 train
事件拟合。prototype cosine difference 不保证以零为分类边界，因此 risk threshold 只在 bank-train 上按
最大 balanced accuracy 冻结，再原样用于 holdout 与在线阶段。holdout qualification 要求：

- train/holdout 的 recovery、persistence 各至少 40 个事件；
- ROC AUC 不低于 `0.60`；
- train-fitted threshold 的 balanced accuracy 与 AP lift 完整记录为诊断，但不作为第二个硬门槛。

旧 boundary gate 的事件位置和“下一 boundary”标签与 V3.4 的逐 token 稳定恢复标签并不相同，因此旧版
`0.8026/0.7180` 不能作为 V3.4 的直接基准。V3.4 的离线硬门槛只回答 hidden state 是否在未见过的
experience 上具有高于机会水平的排序辨别力；最终 gate 是否有任务收益仍由 matched dev 决定。

同一次离线 pass 还记录 vocabulary entropy 与 raw top1-top2 logit margin，但二者只作为诊断，不参与
V3.4 trigger。运行：

```bash
bash scripts/experiments/gsm8k/run_v3_4_token_risk_gate.sh \
  "$PHASE1_DIR" "$OUTPUT_ROOT/v3_4_token_risk"
```

在线阶段压缩为两次验证。第一次在 calibration-val 上跑 64 题、关闭 selector；该日志同时用于验证
current-token pooling/top-1、报告 hubness，并 answer-blind 地按 first-attempt margin 保留最高 50%。
第二次直接跑完整 473 题 matched dev，与已有 V3.1 boundary-last margin 结果逐题比较。每个 gate token
记录 attention entropy、risk、vocabulary entropy、logit margin、low streak、active memory before/after
和 `t+1` effect index；报告另外分开统计 native 与 memory-conditioned risk 分布，避免把 treated-state
risk drift 误解为因果效果。

```bash
bash scripts/experiments/gsm8k/run_v3_4_continuous_gate_experiment.sh \
  "$PHASE1_DIR" \
  "$E0_DIR" \
  "$OUTPUT_ROOT/v3_4_token_risk/token-entropy-risk-gate-v3.4.pt" \
  "$OUTPUT_ROOT"
```

V3.4 offline token-risk qualification 已通过，但 matched dev 对 V3.1 的 strict delta 为
`-0.0105708245`，bootstrap 95% CI 为 `[-0.0232558140, 0.0021141649]`，因此 V3.4 task qualification
未通过。其 go/no-go 工程阈值仍保留为 strict point delta 至少 `0`、strict bootstrap 95% CI 下界至少
`-1.5%`、format point delta 至少 `-0.5%`，但 V3.4/V3.5 final-test 当前都被封锁，旧 runner 的
`--run-final` 不能作为本轮执行指引。

### V3.5 applicability-aware continuous memory

V3.5 是一次 compound revision，同时改变 selector 与 abstain 后的 memory 生命周期：

1. question-only applicability shortlist + shortlist 内 full-prefix dynamic rerank；
2. terminal abstain；若已有 active memory，则清除旧 memory 并 native re-forward 当前 observation token。

第一轮 matched dev 可以一起验证两项变化，但任何改善都不能严格归因到其中一项。若值得继续，应另做
selector-only、terminal-clear-only、selector+terminal-clear ablation；不得根据已知 harmed sample ID
调 threshold，也不得删除或 blacklist `mem-051ae8fcf60f21781c7f145f`。

#### Artifacts 与 fail-closed 绑定

V3.5 使用独立 schema，不让旧 V3.1/V3.4 margin artifact 静默兼容：

- system profile：`experience-memory-system-profile-v3.5`；
- dual bank：`experience-memory-v3.5-dual-key-bank-v1`；
- applicability calibration：`experience-memory-v3.5-applicability-calibration-v1`；
- final selector calibration：`experience-memory-v3.5-selector-calibration-v1`；
- retrieval decision：`experience-memory-v3.5-retrieval-decision-v1`；
- generation result：`experience-memory-v3.5-generation-result-v1`。

dual manifest 逐条绑定 applicability/dynamic text、token IDs、embedding hashes、`memory_id`、
`source_experience_id`、payload hash、KV slot/layer；全局绑定 Phase-1 approved/verified files、MemoryRecord、
原 V3 applicability bank tensor/manifest/offline report、E0 final report、side-KV manifest、reasoner、tokenizer、
layer 24、SDPA、dtype 与 ordered-memory hash。loader 必须拒绝 schema、logical hash、provenance、顺序、
revision 或 task-blind 标志不一致的 artifact。

离线复用还绑定编译时的代码身份，而不只记录可能长期不变的 `compiler_git_revision`：dual manifest 的
`input_artifacts`、offline report 的 `inputs` 与 applicability calibration 的 `source` 必须包含且一致地绑定
19 个 V3.5 实现文件的逐文件 SHA、该 map 的 canonical set SHA，以及仅覆盖这些实现路径的 tracked diff
SHA。runner 每次复用前按 compiler 的同一 scoped-diff 算法重算并逐项比较，同时重算 split manifest 的
logical SHA、dataset revision、dual manifest/report/calibration logical hash；任一代码或输入漂移都停止。

runner 的输出规格为：

```text
$OUTPUT_ROOT/v3_5_applicability_selector/
├── dual_key_bank/
│   ├── dual_retrieval_key_bank.safetensors
│   ├── dual_retrieval_key_manifest.json
│   ├── offline_report.json
│   └── offline_report.md
├── applicability_calibration.json
├── applicability_calibration.md
├── calibration_trace/
│   ├── results.jsonl
│   ├── run_profile.json
│   ├── run_report.json
│   └── query_embeddings/                 # 每个有 attempt 的样本一个 safetensors sidecar
├── selector_calibration.json
├── selector_calibration.md
├── dev/
│   ├── results.jsonl
│   ├── run_profile.json
│   ├── run_report.json
│   ├── analysis_report.json
│   └── analysis_report.md
├── dev_v35_minus_v34.json
├── dev_v35_minus_v34.md
├── dev_v35_minus_v31.json              # 兼容 baseline 存在时
├── dev_v35_minus_v31.md                # 兼容 baseline 存在时
├── dev_qualification.json              # V3.4 comparison 存在时
└── dev_qualification.md
```

offline report 与两个 calibration artifact 都必须是 `status=passed`，并显式记录
`task_accuracy_used=false`、`answer_or_reward_used=false`；否则后续阶段停止。calibration trace 使用独立
trace-only profile，固定 `calibration-val --limit 64`，只提供 first-attempt shortlist margin。dynamic
threshold 以 inclusive tie policy 保留约 50%，不能读取 completion correctness、answer 或 reward。Stage B
显式启用 `--save-query-embeddings`；sidecar representation 固定为
`dynamic_query_l2_normalized_exact_audit`，逐 attempt 保存 retrieval 真正使用的 L2-normalized float32
原始 bits。calibrator 必须认证逐行 sidecar 文件 hash、attempt 集合以及 `attempt_01` tensor 的
embedding hash/norm 与 retrieval decision 一致。正式 473 题 dev 不保存该 sidecar。

#### Static selector、dynamic rerank 与状态机

static query 精确为 `question.strip()`，通过 `tokenizer.encode(..., add_special_tokens=False)` 编码；它不含
ChatML wrapper 或统一 prompt。全库 applicability cosine 先按 score 降序、再按 `memory_id` 稳定打破
tie，应用冻结 floor 后保留 top-k。shortlist 对整次 generation 不变，dynamic retrieval 只能返回其中 ID。

dynamic query 仍是 canonical prompt + 当前 token 在内的完整 partial CoT，side-KV 暂停，layer-24
current-generated-token pooling，L2 normalize。只在 shortlist 的 dynamic embeddings 中计算 exact cosine
top-2；admission 同时要求 static score ≥ floor、dynamic margin ≥ threshold、ID 属于 shortlist、KV metadata
完全对齐。V3.5 不增加 dynamic absolute-score gate。

| decision | active memory before | outcome / memory after | gate after |
|---|---|---|---|
| selected new ID | none | activated / selected ID | `DISARMED`，第 3 次则 `EXHAUSTED` |
| selected different ID | old ID | replaced / selected ID | `DISARMED`，第 3 次则 `EXHAUSTED` |
| selected same ID | same ID | duplicate / same ID；attempt 仍消耗 | `DISARMED`，第 3 次则 `EXHAUSTED` |
| selector abstain | none | terminal abstain / none | `EXHAUSTED`，永不 re-arm |
| selector abstain | old ID | terminal abstain + deactivated / none | `EXHAUSTED`，永不 re-arm |

最后一行不能只调用 `controller.deactivate()`：old-memory probe 是 counterfactual，必须截断其 attention
trace、恢复 probe 前 native cache length、deactivate、清空 `current_memory`，再在 side-KV inactive 状态
用当前 token forward，并用该 native output 生成 `t+1`。日志用
`deactivated_on_terminal_abstain` transition、`cleared_memory_id`、`clear_affects_generated_token_index`、
deactivation KL/top1/timing 和 `actual_path_after_abstain=native` 证明没有 stale-memory effect。

#### 一键 runner 与正常停止点

```bash
bash scripts/experiments/gsm8k/run_v3_5_applicability_selector_experiment.sh \
  --v3-bank-dir "$V3_BANK_DIR" \
  --v34-dev-dir "$V34_DEV_DIR" \
  --v31-dev-dir "$V31_DEV_DIR" \
  "$PHASE1_DIR" \
  "$E0_DIR" \
  "$TOKEN_RISK_ARTIFACT" \
  "$OUTPUT_ROOT"
```

四个 positional 依次是 Phase-1 artifact 目录、E0 artifact 目录、已通过资格验证的 V3.4 token-risk
artifact、实验输出根目录。三个 optional baseline 目录允许复用原 V3 bank、V3.4 matched dev 和 V3.1
matched dev；缺少某个 comparison baseline 时只跳过对应 comparison，不伪造报告。已认证的 dual bank、
calibration trace 与 dev run 会复用，任何已有工件的 schema/provenance/profile 不匹配都会停止。Stage B/C
复用并非只看文件是否存在：runner 用 evaluator 的同一 logical-hash 实现重算 `profile_sha256`，逐项比对
当前 git revision、tracked diff 与实现文件集合；再认证 `results.jsonl` 的 row schema/profile hash/row hash、
唯一且有序的 sample IDs、completion token count/hash，并从这些行重算 summary 与 `run_report.json` 的
logical hash。代码变化、行篡改、重复/缺失样本或自洽性失败都不能被当作已完成 run 复用。

正常停止点永远是 473 题 exploratory matched dev 的 analysis/comparison/qualification。qualification 的
硬完整性条件包括 exact zero-attempt/static-unavailable parity、outside-shortlist/KV/stale-attention/budget/
re-arm violations 全为零，以及 calibration task-blind；任务条件以 V3.5 minus V3.4 的 strict delta、strict
CI lower 和 format delta 为主。无论结果是否通过，`qualified_for_final_test` 都保持 `false`，runner 不含
final-test 执行路径。

#### Dynamic source-state alignment diagnostic

当 static applicability qualification 未通过或需要单独检查第二层 dynamic rerank 时，使用独立诊断：

```bash
bash scripts/experiments/gsm8k/run_v3_5_dynamic_source_alignment_audit.sh \
  "$PHASE1_DIR" \
  "$E0_DIR" \
  "$OUTPUT_ROOT/v3_4_token_risk/token-entropy-risk-gate-v3.4.pt" \
  "$OUTPUT_ROOT"
```

该 audit 明确绕过 static shortlist，直接在全部 161 条已认证 dynamic keys 上检索；它不会改变或绕过
V3.5 offline `not_qualified` 状态。每条 memory 通过 `source_experience_id` 找回同题的 verified-success
`trajectory` 与 verified-failure `reference_trajectory`，沿用 GSM8K prompt contract、完整 partial CoT、
layer-24 `current_generated_token` pooling、L2 与 side-KV disabled 合同。两条 source trajectory 的每个
pre-answer token 都产生 own-memory rank curve；primary 位置固定为 failure/reference trajectory 上冻结
V3.4 entropy+risk gate 的第一次 counterfactual joint-qualified event，不允许按 retrieval score 事后挑位置。

报告包含 reference/target first-gate Recall@1/5/10/32、MRR、rank 分布、attempt 2/3、全 token macro
coverage、own-key cosine 与 best-other gap、target/reference 配对 rank 差、top-1 hubness、稳定
tie-break top hits，以及保持完整 score geometry 的 query→own-memory ID 置换 null。正式 anchor 另做
independent exact full-prefix re-encode，并把 query embeddings 写入认证的
safetensors sidecar。整个诊断不运行生成、不加载 side-KV，也不把任务 accuracy、answer 或 reward 用于
query ranking、threshold 或结果位置选择；它只复用 bank 已认证的 success/failure source 角色，不产生
新的 selector threshold。低 source-state recall 可以否定当前 abstract dynamic key 与 runtime prefix 的
基本对齐；高 recall 只表示必要的 in-source sanity check 通过，不能证明跨题泛化或任务收益。若 target
明显强而 reference 弱，应解释为 confirmation retrieval，而不是 corrective retrieval。

### Injection layer

当前 V3 只在 layer 24 注入，不做 layer search、multi-layer 或候选层双路编译。等 V3 全流程和全量
结果完成后，再单独研究校准过的 injection layer；届时必须产生新的 layer-specific key/KV 工件和
profile，不能原地修改本版 layer-24 结果。
