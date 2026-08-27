# MemGen V3：当前系统与执行流程

V3 冻结为 layer-24、embedding key → compiled side-KV value、entropy hysteresis re-arm 的系统。
V2 工件和入口仍保留用于历史复现，但新的实验必须使用 V3 schema 和脚本。

## 1. 完整数据流

```text
离线阶段
MemoryRecord
  ├─ sanitized when_facing
  │    └─ frozen reasoner layer-24 / last token / L2
  │          └─ embedding key bank
  └─ full When facing + Prefer + Avoid
       └─ frozen reasoner layer-24 canonical pre-RoPE
             └─ side-KV value bank

在线阶段
question + full generated partial CoT（从首个生成 token 到当前 boundary）
  └─ side-KV 暂停，pure-prefix 全量重编码
       └─ exact cosine top-2（top-1 选择，top-2/margin 记录；当前不设 abstain threshold）
            └─ memory_id 对齐到已编译 side-KV
                 └─ layer-24 注入；新 memory 替换旧 memory
```

key 与 value 通过同一个 `memory_id`、`payload_hash` 和固定 record order 绑定。payload 与 provenance
仍保留用于审计，但线上不再用 BM25 文本匹配，也不在线重新编译 KV。

## 2. 离线阶段

V3 不重新生成 Teacher/Pro 经验，也不重新搜索注入层。它复用已经通过 E0 的 161 条
`MemoryRecord` 和 layer-24 canonical side-KV，只新增一次 retrieval-key 编译与交叉资格校验：

1. 验证 `e0_final_report`、records 和 side-KV 是同一套正式 E0 工件。
2. 只取 `sanitized_fields.when_facing` 作为检索 key 的语义来源。
3. 用相同冻结 reasoner 在 hidden-state tuple index 24 取 last-valid-token state，转 float32 并 L2
   normalize。
4. 保存 `retrieval_key_bank.safetensors` 和带逐条 embedding hash/norm 的 manifest。
5. 验证 text/key/KV 的 ID、顺序、payload hash、layer、reasoner revision 全部一致，产生一次
   `v3_offline_report.json`。

入口：

```bash
python scripts/compile_v3_retrieval_keys.py \
  --memory-records "$E0_DIR/memory_records.v2.jsonl" \
  --side-kv-manifest "$E0_DIR/side_kv_manifest.json" \
  --e0-final-report "$E0_DIR/e0_final_report.json" \
  --output-dir "$V3_BANK_DIR" \
  --device cuda \
  --dtype bfloat16
```

当前不再做“双路编译”。entropy-risk artifact 也不属于 memory value 编译；它只提供已校准的高/低熵
阈值和 recovery/persistence 诊断中心。

## 3. 在线阶段

### Trigger 状态机

初态为 `ARMED`，只检查答案前的 `,`、`.`、换行 boundary：

- `ARMED + entropy >= high`：触发一次 retrieval attempt。
- 每次 attempt（包括 duplicate 或 abstain）都消耗预算，并进入 `DISARMED`；第三次后进入
  `EXHAUSTED`。
- `DISARMED + entropy <= low`：只执行 re-arm。这个低熵 boundary 本身禁止再次触发；必须等待未来的
  新 high-entropy boundary。
- 出现 answer marker 或 EOS 后进入 `CLOSED`。
- persistence-risk score 在每个被检查的 boundary 记录，但不参与 trigger 或 re-arm 判断。

每题最多 3 次 retrieval attempt。首次选中 memory 时激活；后续选中不同 memory 时替换当前
memory；选中相同 memory 时记为 duplicate，旧 memory 保持激活。系统始终最多只有一个 active
memory，不累积 K/V。

### Cache 与 query 隔离

- 检索 query 是 canonical prompt token IDs 加当前为止全部生成 token IDs，包括当前 delimiter。
- query encoder 从头重算 prefix，期间 side-KV 强制暂停；因此检索不使用 memory-conditioned query
  state，也不读取 live cache。
- 发生替换时，旧 memory 与新 memory 都从“处理当前 boundary 之前”的同一份 native KV cache
  分支计算。选择新 memory 后只保留新分支。
- side-KV 始终走 layer-24 side path，不写入 Hugging Face native KV cache；active memory 持续到
  replacement 或 EOS。

单题入口：

```bash
python scripts/run_online_experience_memory_v3.py \
  --question "..." \
  --memory-records "$E0_DIR/memory_records.v2.jsonl" \
  --retrieval-key-manifest "$V3_BANK_DIR/retrieval_key_manifest.json" \
  --side-kv-manifest "$E0_DIR/side_kv_manifest.json" \
  --v3-offline-report "$V3_BANK_DIR/v3_offline_report.json" \
  --e0-final-report "$E0_DIR/e0_final_report.json" \
  --risk-artifact "$RISK_ARTIFACT" \
  --output "$OUTPUT_JSON" \
  --device cuda \
  --dtype bfloat16
```

## 4. 评估与日志

正式比较只有 `vanilla` 和 `v3`。任务汇总只包含：

- 严格准确率：仓库官方 GSM8K first-boxed reward。
- 格式准确率：第一个 boxed answer 存在且可解析。
- 生成 token：逐题计数（包含首次 EOS），以及 total/mean/median/p95/max 和配对差值。

runner 每题 append、flush、`fsync`，并在 resume 时验证 profile 和每行 hash。日志包含：

- run provenance、工件 hash、reasoner/tokenizer revision、prompt/generation profile；
- 每个 boundary 的 gate state、entropy、risk、动作和 active memory；
- 每次 attempt 的完整 query token/hash、embedding hash/norm、top-2 score/margin、latency 和
  duplicate/replacement/abstain；
- memory transition/span、逐步 attention mass、native-cache length、替换首步 logits KL；
- completion/token IDs、严格/格式结果、token 数和 runtime；
- attempt/re-arm/replacement/duplicate/memory exposure 汇总与 cache integrity。

默认不保存 full logits 或 hidden states；可用 `--save-query-embeddings` 保存允许的 query embedding
sidecar。

```bash
python scripts/evaluate_v3_experience_memory.py \
  --split-manifest "$PHASE1_DIR/split_manifest.json" \
  --logical-split calibration-val \
  --memory-records "$E0_DIR/memory_records.v2.jsonl" \
  --retrieval-key-manifest "$V3_BANK_DIR/retrieval_key_manifest.json" \
  --side-kv-manifest "$E0_DIR/side_kv_manifest.json" \
  --v3-offline-report "$V3_BANK_DIR/v3_offline_report.json" \
  --e0-final-report "$E0_DIR/e0_final_report.json" \
  --risk-artifact "$RISK_ARTIFACT" \
  --output-dir "$RUN_DIR" \
  --limit 8 \
  --device cuda \
  --dtype bfloat16
```

全量评估结束后，可在不加载模型的 CPU 环境中对逐题日志做配对诊断：

```bash
python scripts/analyze_v3_evaluation.py \
  --results "$RUN_DIR/results.jsonl" \
  --run-profile "$RUN_DIR/run_profile.json" \
  --output "$RUN_DIR/analysis_report.json" \
  --markdown-output "$RUN_DIR/analysis_report.md"
```

分析器会复核 profile/逐行/completion hash 与 V3 在线不变量，并输出 strict/format 的配对四格表、
McNemar 检验、bootstrap 区间、零触发 parity、多次 attempt/replacement 分层、检索分数、KL、
attention mass、memory ID 和 token 尾部异常。Markdown 是结论摘要，JSON 保留完整诊断与样本 ID；
其中 memory/相关性/分位数组只作为探索性结果，不作为独立最终确认。

## 5. 压缩实验顺序

1. 离线只跑一次 key/KV 对齐资格验证。
2. 在线先跑一次很小的 calibration smoke；需要时再跑一次稍大的 calibration validation。
3. 日志与 integrity 全部通过后，直接跑 `--logical-split final-test --limit 0` 的 1319 题
   vanilla-vs-V3 全量比较。

official test 已在先前研究中使用，因此新一轮 `final-test` 必须标为
`reused_official_test_descriptive_evaluation`，不能宣称为独立最终确认。runner 会把这个限制写进
profile 和 final report。

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

### Injection layer

当前 V3 只在 layer 24 注入，不做 layer search、multi-layer 或候选层双路编译。等 V3 全流程和全量
结果完成后，再单独研究校准过的 injection layer；届时必须产生新的 layer-specific key/KV 工件和
profile，不能原地修改本版 layer-24 结果。
