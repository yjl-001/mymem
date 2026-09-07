# 实验入口

仓库只保留当前有效的经验记忆链路和原始 MemGen 训练入口。历史负向实验可从 Git 历史恢复，不再保留
可执行 runner，避免误用已关闭路线。

## 服务器配置

Phase 1 使用共享环境文件：

```bash
cp scripts/experiments/server.env.example scripts/experiments/.server.env
```

E0/E1 分别使用最小环境文件，其中只配置输出根目录和可选 GPU：

```bash
cp scripts/experiments/gsm8k/e0.server.env.example \
  scripts/experiments/gsm8k/.e0.server.env
cp scripts/experiments/gsm8k/e1.server.env.example \
  scripts/experiments/gsm8k/.e1.server.env
```

## 当前经验记忆流程

### 优先：从 semantic packet 零付费恢复当前 V4 source evidence

如果下面的文件仍存在：

```text
offline/construction_v4_2_semantic/semantic_evidence_packets.jsonl
```

当前 curated 17-bank 使用的 116 条原题、success/failure 原始轨迹和 verifier 证据可以直接恢复，不需要调用
DeepSeek，也不需要重新生成 bank 或 Side-KV。服务器 smoke：

```bash
bash scripts/experiments/gsm8k/run_v4_question_recovery.sh \
  --mode smoke \
  --stage all \
  gsm8k-v4-packet-replay-20260907-r1 \
  /data/memgen-runs/v4/offline/construction_v4_2_semantic/semantic_evidence_packets.jsonl \
  /data/memgen-runs/v4/offline/construction_v4_2_local_curated \
  /data/memgen-runs/v4/offline/side_kv_v4_2_local_curated \
  /data/memgen-runs
```

smoke 通过后用同一个 `RECOVERY_ID` 运行 full；risk 会复用，cache/oracle 写入独立的 full 目录：

```bash
bash scripts/experiments/gsm8k/run_v4_question_recovery.sh \
  --mode full \
  --stage all \
  gsm8k-v4-packet-replay-20260907-r1 \
  /data/memgen-runs/v4/offline/construction_v4_2_semantic/semantic_evidence_packets.jsonl \
  /data/memgen-runs/v4/offline/construction_v4_2_local_curated \
  /data/memgen-runs/v4/offline/side_kv_v4_2_local_curated \
  /data/memgen-runs
```

该入口只访问公开 GSM8K 和本地 Qwen，显式清除付费 provider keys。新 risk 使用全部 116 条 packet replay
轨迹并保持正式 qualification 门槛；smoke 只缩小 cache/oracle。输出明确声明它不是旧 Phase-1 文件或旧 risk
artifact 的 byte-identical 恢复，也不是 held-out 泛化实验。

### 备选：全新 Phase-1 / risk 数据与唯一血缘

仅当 semantic packet 也已丢失，并且明确接受新的付费 teacher/reviewer 调用时，才使用本节。Phase-1 与 risk
不再建议以两个散落目录传给后续实验。需要重新构造时，先选一个永久不复用的
`LINEAGE_ID`，一次性写入固定根目录：

```bash
bash scripts/experiments/gsm8k/run_phase1_risk_lineage.sh \
  --stage all \
  --allow-paid-phase1 \
  --bank-manifest /data/memgen-runs/v4/offline/construction_v4_2_local_curated/bank_manifest.json \
  --side-kv-manifest /data/memgen-runs/v4/offline/side_kv_v4_2_local_curated/v4_side_kv_manifest.json \
  gsm8k-v4-phase1-20260907-r1 \
  /data/memgen-runs
```

`--allow-paid-phase1` 是必需的显式确认，因为完整 Phase-1 会执行 teacher/reviewer API 调用；risk
阶段本身不调用外部教师。脚本不会进入 V4 selector、source-state、oracle、dev-test 或 final-test。

成功后只使用该 lineage 内生成的环境文件，不再手填两个路径：

```bash
source /data/memgen-runs/lineages/gsm8k/gsm8k-v4-phase1-20260907-r1/USE_THIS_LINEAGE.env
```

同目录的 `phase1_risk_lineage_manifest.json` 绑定完整 Phase-1、V3.4 risk evidence/report/artifact
的 SHA，并记录其与指定 V4 bank/Side-KV 的逐项兼容性。sealed lineage 不允许原地改写；任何输入变化都必须
换新的 `LINEAGE_ID`。

特别注意：重新采样得到的新 Phase-1 通常**不兼容**旧 V4 bank。旧 bank 绑定的是原
`verified_experiences.jsonl` 和 `split_manifest.json` 的文件级 SHA；即使配置和样本 ID 相同，也不能把
新文件当成原文件。manifest 若报告
`incompatible_rebuild_or_original_data_recovery_required`，必须恢复原始 Phase-1，或从新 lineage 重建
下游 bank/Side-KV；不能直接运行 V4 cache/oracle。

1. 构造 verifier-backed Phase 1 bank：

```bash
bash scripts/experiments/gsm8k/run_phase1_verified_bank.sh
```

2. 在 SDPA 下重新编译冻结 entropy-risk gate artifact：

```bash
bash scripts/experiments/gsm8k/run_entropy_risk_gate.sh "$PHASE1_DIR"
```

风险 artifact 与 prompt contract、attention backend 绑定。旧 eager artifact 不能与当前 SDPA runtime
混用，必须重新编译；这一步不调用 Teacher/Pro。

3. 构造并在 SDPA 下审计 MemoryRecord 与 canonical side-KV：

```bash
bash scripts/experiments/gsm8k/run_e0_experience_memory.sh "$PHASE1_DIR"
```

4. 验证 canonical base reasoner 与 E1 live-cache runtime 对齐：

```bash
bash scripts/experiments/gsm8k/run_base_reasoner_parity.sh \
  --logical-split final-test \
  --limit 32 \
  "$PHASE1_DIR" "$E0_DIR"
```

`base_parity_summary.json` 同时报告仓库原始 `inputs_embeds + use_cache=False`
HuggingFace greedy 与显式 KV-cache greedy 的严格准确率、诊断准确率和逐 token parity。正式 E1 前要求
`exact_token_parity=true`；正式 GSM8K 生成预算固定为 1024。

若需隔离 attention backend，使用同一命令运行任意两个受支持的 backend：

```bash
MEMGEN_RUN_TAG=base-attention-final32-v1 \
bash scripts/experiments/gsm8k/run_base_attention_backend_comparison.sh \
  --logical-split final-test \
  --limit 32 \
  --reference-backend eager \
  --candidate-backend flash_attention_2 \
  "$PHASE1_DIR" "$E0_DIR"
```

该诊断固定 `batch_size=1`、题目、prompt token、模型 revision、dtype 和 decoding，只改变
`attention_implementation`。`comparison_summary.json` 报告两个 backend 各自的 native/cache parity、
准确率差以及跨 backend 的逐 token 分叉；不运行 gate、检索或 memory 注入。

已确认 eager 会显著破坏当前 reasoner 后，使用同一诊断完成 SDPA 检查：

```bash
MEMGEN_RUN_TAG=base-sdpa-final32-v1 \
bash scripts/experiments/gsm8k/run_base_attention_backend_comparison.sh \
  --logical-split final-test \
  --limit 32 \
  --reference-backend flash_attention_2 \
  --candidate-backend sdpa \
  "$PHASE1_DIR" "$E0_DIR"
```

当前 32 题诊断中 SDPA strict accuracy 为 `0.53125`，且 native/cache 逐 token 一致；正式 E1
固定使用 SDPA。FlashAttention2 仍是质量参考，不与 SDPA 系统效果混称。

5. 运行 gate、BM25 和 persistent side-KV 完整评测：

```bash
bash scripts/experiments/gsm8k/run_e1d_full_system.sh \
  --logical-split calibration-val \
  --limit 100 \
  "$PHASE1_DIR" "$E0_DIR" "$RISK_ARTIFACT"
```

明确决定执行官方 GSM8K test 的全量冻结评测时：

```bash
bash scripts/experiments/gsm8k/run_e1d_full_system.sh \
  --logical-split final-test \
  --limit 0 \
  "$PHASE1_DIR" "$E0_DIR" "$RISK_ARTIFACT"
```

`--limit 0` 表示运行整个 logical split；final-test artifact 会显式标记为 `final_evaluation`。

E1 只比较 `vanilla` 和 `matched`；gate observation 仅作为冻结触发位置与检索输入的内部审计路径，
不作为测评条件。完整结论、系统契约与解释限制见
[`experience_calibrated_steering_plan.md`](../../docs/codex/experience_calibrated_steering_plan.md)。

## V3.1 margin selector 实验

V3.1 复用已有 layer-24 embedding/side-KV bank，只对 exact-cosine top-2 选择增加由
calibration-val answer-blind 冻结的 margin abstention。评测 runner 使用：

```bash
bash scripts/experiments/gsm8k/run_v3_1_selector_experiment.sh \
  --calibration-limit 0 \
  --dev-limit 0 \
  --target-retained-fraction 0.5 \
  "$PHASE1_DIR" "$E0_DIR" "$RISK_ARTIFACT" "$OUTPUT_ROOT"
```

完整的 key geometry 审计、calibration artifact 构造和 matched baseline/V3.1 比较命令见
[`memgen_v3_system.md`](../../docs/codex/memgen_v3_system.md)。本轮不使用 final-test 调阈值，也不修改
注入层。
