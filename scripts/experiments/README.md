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
同题 contrast、调用离线 teacher、执行自动 quality gate，并生成 30 条人工复核清单。
`approved_bank_records.jsonl` 只包含绑定了真实 `verified_failure` episode 的记录。
