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
同题 contrast、调用 Flash teacher、执行确定性审计，再由 Pro reviewer 独立复核。
确定性审计只对 provenance、verifier 绑定和 schema 等数据完整性负责；这类异常不参与
内容优劣判断，而是单独进入 `quarantined_bank_records.jsonl` 等待修复或重建。关键词、
相似度、格式描述和 teacher 自评等都只是语义警告，最终内容通过或拒绝由 Pro 的逐字段
证据审核决定。八个 bank 字段和五个 pair 属性全部 supported 才进入正式 bank；
存在 unsupported/contradicted 时拒绝，其余包含 partially supported 的记录进入
`deferred_bank_records.jsonl`。Reviewer confidence 仅作诊断，不参与路由。报告同时按
`experience_type` 统计各路由数量与通过率，用于检查精度优先筛选是否造成类别覆盖偏差。

Phase 1 verifier 同时保存严格任务奖励和诊断字段。缺少或损坏 `\\boxed{}` 仍然是
正式任务失败；若宽松诊断能确认自然语言中的最终数值正确，该 reference 会被归为
`format_compliance`，而不是被误写为推理失败。答案错误和混合/无法判定失败使用独立
experience type，供后续分簇编译。Teacher prompt 会收到 target/reference 两侧的完整
verifier 记录。确定性审计会检查失败类型一致性，并把 format-specific 描述问题作为
语义警告交给 Pro 判断。

脚本会复用已经完成的 student rollout，但每次都会廉价重建 typed experiences。旧版
Teacher/Reviewer 记录、模型或 provenance 已变化的记录会在 `--resume` 时自动丢弃并重新
生成，因此升级后无需重新执行 GPU rollout。审核路由规则变化时，来源与模型未变化的
首轮 Pro 结果会复用并重新路由。被淘汰的旧 JSONL 会先保存为同目录下带
`.stale-<UTC>.bak` 后缀的备份。

`ai_approved_bank_records.jsonl` 即正式 Phase 1 bank 输入；`ai_rejected`、`deferred`
和 `quarantined` 仅保留用于审计、误差分析或未来扩充稀缺类别。这里的 `AI review` 不得
在论文或实验报告中表述为人工审核。

Phase 2 会从正式 bank 回连到 `verified_experiences.jsonl` 的原始 student 轨迹，构造
多个层的全局 residual vector；Teacher 文本不直接参与 hidden-state 差分。它先在
`calibration-val` 的 tune 子集校准 layer、alpha、soft-gate slope 与注入预算，随后在
同一 split 的独立 confirmation 子集运行 vanilla、entropy-only、真实 vector、随机
boundary、同范数随机 vector 和反转 vector 六个对照。运行时只需传入冻结的 Phase 1
目录：

```bash
bash scripts/experiments/gsm8k/run_phase2_global_steering.sh \
  /absolute/path/to/gsm8k_phase1_verified-student-contrast_<tag>
```

初始默认使用 100 个 tune + 100 个 confirmation 样本，避免在发现架构/环境错误前占用
整段 GPU 时间。成功后可仅在服务器 `.server.env` 中提高 `MEMGEN_PHASE2_TUNE_SIZE` 和
`MEMGEN_PHASE2_CONFIRM_SIZE`；不要改动已提交脚本。正式 vector 只使用
`answer_correctness` evidence，`format_compliance` 等类型会同时单独编译，留给后续
条件化/类型化实验，避免格式监督淹没主推理向量。
