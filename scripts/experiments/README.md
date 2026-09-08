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

当前服务器总入口 `test.sh` 默认运行 **V4.3 Unified Heuristic Memory**：

```bash
git pull
./test.sh
```

不带参数时，使用 DeepSeek 综合统一卡片，再执行 primary/conditional Side-KV 编译、四层 smoke，认证通过后执行 full。
首次构造需要环境中的 `DEEPSEEK_API_KEY`；每 Bank 一次正常请求，完整缓存复用不请求 API。
当前 prompt v3 将 17 个 Bank 全部重新生成，不带入旧卡片或旧响应；只复用同版中断缓存。
内容不再用数字、公式、实体或关键词静态拦截，改用教师提示词约束适用条件与过度概括；
响应结构、证据身份、支持数与工件完整性检查保留。
默认使用 `/data/memgen-runs` 及 recovery lineage `gsm8k-v4-packet-replay-20260907-r1` 中已经存在的
完整 116-sample source cache 和 risk；缺失旧工件直接停止，不自动恢复或重新拟合。
可用 `./test.sh construct` 只生成卡片，不要求 GPU/source cache/risk；也可用 `./test.sh smoke`、
`./test.sh full` 或 `MEMGEN_V43_VALIDATE_ONLY=1 ./test.sh all`。
新输出使用 `*_v4_3_deepseek_prompt_v3` 独立目录，支持按 Bank 响应和按 case 认证恢复；
脚本在构造结束后清除 DeepSeek key，编译和审计保持 offline-only。
配置与工件说明见 [V4.3 完整实现合同](../../docs/codex/memgen_v4_3_construction.md)。

旧 V4.2 自动选择 `stage=oracle/all` 的行为保留为显式 `./test.sh legacy [smoke|full|all]`。
下面的 recovery/旧 oracle 命令属于该历史流程，不由 V4.3 自动调用。

下面保留等价的底层命令，便于排错或精确控制单个阶段。服务器 smoke：

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

Oracle 的 full-answer profile 保持 Side-KV 最多 active 32 token，但 memory 卸载后继续原生生成，直到完整
`\boxed{}`、EOS 或总 completion 达到 1024 token。它把 local-32 和 final-outcome 指标分开写入
`oracle_audit_full_answer/`，不会覆盖早期 `oracle_audit/` 中的 32-token 局部报告。

如果相同 `RECOVERY_ID` 已完成旧版 `--stage all`，只需重跑 oracle，不要重新恢复、拟合 risk 或提取 cache。先跑
smoke：

```bash
bash scripts/experiments/gsm8k/run_v4_question_recovery.sh \
  --mode smoke \
  --stage oracle \
  gsm8k-v4-packet-replay-20260907-r1 \
  /data/memgen-runs/v4/offline/construction_v4_2_semantic/semantic_evidence_packets.jsonl \
  /data/memgen-runs/v4/offline/construction_v4_2_local_curated \
  /data/memgen-runs/v4/offline/side_kv_v4_2_local_curated \
  /data/memgen-runs
```

通过后复用 full cache：

```bash
bash scripts/experiments/gsm8k/run_v4_question_recovery.sh \
  --mode full \
  --stage oracle \
  gsm8k-v4-packet-replay-20260907-r1 \
  /data/memgen-runs/v4/offline/construction_v4_2_semantic/semantic_evidence_packets.jsonl \
  /data/memgen-runs/v4/offline/construction_v4_2_local_curated \
  /data/memgen-runs/v4/offline/side_kv_v4_2_local_curated \
  /data/memgen-runs
```

两次运行都不会启动 selector、dev-test、final-test 或外部教师 API。

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
