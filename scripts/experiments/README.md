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

1. 构造 verifier-backed Phase 1 bank：

```bash
bash scripts/experiments/gsm8k/run_phase1_verified_bank.sh
```

2. 编译冻结 entropy-risk gate artifact：

```bash
bash scripts/experiments/gsm8k/run_entropy_risk_gate.sh "$PHASE1_DIR"
```

风险 artifact 与 prompt contract 绑定。旧 artifact 使用过带额外换行的 prompt，
不能与当前 canonical baseline 混用，必须重新编译；这一步不调用 Teacher/Pro。

3. 构造并审计 MemoryRecord 与 canonical side-KV：

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
