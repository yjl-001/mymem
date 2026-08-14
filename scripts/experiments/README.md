# 一键实验脚本

每个正式训练或评测都必须同时包含：

1. `configs/experiments/<dataset>/<method>.yaml`：提交到 Git 的参数差异；
2. `scripts/experiments/<dataset>/run_<method>.sh`：提交到 Git 的一键启动脚本。

脚本负责：读取服务器本地环境、设置明确的 `run-id`、调用
`scripts/launch_experiment.py`。`run-id` 使用
`<dataset>_<method>_<key-settings>_<stage>_<timestamp>`，让输出目录能直接说明
实验核心。

## 一次性的服务器设置

```bash
cp scripts/experiments/server.env.example scripts/experiments/.server.env
```

编辑 `.server.env` 中的输出根目录、GPU 与 checkpoint 路径。它只包含服务器本地信息，
已被 Git 忽略。

之后每次运行只需同步代码并执行对应脚本：

```bash
git pull --ff-only origin main
bash scripts/experiments/gsm8k/run_entropy_calibration_sink4_q85.sh
```

需要依赖前一阶段产物时，后续脚本接受该产物的绝对路径作为参数，而不会猜测“最新”运行。

离线 bank 构造同样使用版本化的一键脚本。当前 GSM8K 预览见
[`gsm8k/build_teacher_bank_preview.sh`](gsm8k/build_teacher_bank_preview.sh)；它需要
未提交的 `DEEPSEEK_API_KEY`，且输出仅写入服务器的 `MEMGEN_OUTPUT_ROOT`。

正式的 verifier-backed Phase 1 bank 使用：

```bash
bash scripts/experiments/gsm8k/run_phase1_verified_bank.sh
```

它会依次冻结 split、采样 student rollout、用 GSM8K verifier 标记成功/失败、形成
同题 contrast、调用 Flash teacher、执行确定性 quality gate，再由 Pro reviewer 独立
复核。确定性门与 Pro 高置信一致的记录自动通过或拒绝；只有结论冲突、证据含混或低置信
记录进入 `human_controversy_review.jsonl`。

Phase 1 verifier 同时保存严格任务奖励和诊断字段。缺少或损坏 `\\boxed{}` 仍然是
正式任务失败；若宽松诊断能确认自然语言中的最终数值正确，该 reference 会被归为
`format_compliance`，而不是被误写为推理失败。答案错误和混合/无法判定失败使用独立
experience type，供后续分簇编译。Teacher prompt 会收到 target/reference 两侧的完整
verifier 记录，自动质量门会检查失败类型一致性和 format-specific 描述。

脚本会复用已经完成的 student rollout，但每次都会廉价重建 typed experiences。旧版
Teacher/Reviewer 记录、模型或 provenance 已变化的记录会在 `--resume` 时自动丢弃并重新
生成，因此升级后无需重新执行 GPU rollout。被淘汰的旧 JSONL 会先保存为同目录下带
`.stale-<UTC>.bak` 后缀的备份。

争议记录完成最小人工裁决后合并最终 bank：

```bash
python scripts/finalize_phase1_disputes.py \
  --ai-approved "$RUN_DIR/ai_approved_bank_records.jsonl" \
  --ai-rejected "$RUN_DIR/ai_rejected_bank_records.jsonl" \
  --human-review "$RUN_DIR/human_controversy_review_reviewed.jsonl" \
  --final-approved "$RUN_DIR/final_approved_bank_records.jsonl" \
  --final-rejected "$RUN_DIR/final_rejected_bank_records.jsonl" \
  --report-output "$RUN_DIR/phase1_final_review_report.json"
```

这里的 `AI review` 不得在论文或实验报告中表述为人工审核；人工只负责争议分流中的最终
裁决。
