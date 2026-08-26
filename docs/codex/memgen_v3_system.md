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

当前 V3 只在 layer 24 注入，不做 layer search、multi-layer 或候选层双路编译。等 V3 全流程和全量
结果完成后，再单独研究校准过的 injection layer；届时必须产生新的 layer-specific key/KV 工件和
profile，不能原地修改本版 layer-24 结果。
