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

3. 构造并审计 MemoryRecord 与 canonical side-KV：

```bash
bash scripts/experiments/gsm8k/run_e0_experience_memory.sh "$PHASE1_DIR"
```

4. 运行 gate、BM25 和 persistent side-KV 完整评测：

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

E1 只包含 `vanilla`、`gate_observation_only` 和 `matched_persistent_memory`。完整结论、系统契约与解释
限制见 [`experience_calibrated_steering_plan.md`](../../docs/codex/experience_calibrated_steering_plan.md)。
