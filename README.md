# MemGen

MemGen 让 Agent 在推理流中插入 latent memory，而不更新 reasoner 参数或依赖外部文本
数据库。当前仓库的正式实验采用“版本化 YAML 配置 + 一键 Bash 脚本”的工作流：代码、
配置和脚本提交到 Git；模型、数据集、日志和结果保留在服务器磁盘。

详细实验约定见 [docs/experiment_workflow.md](docs/experiment_workflow.md)，脚本说明见
[scripts/experiments/README.md](scripts/experiments/README.md)。

## 服务器首次设置

```bash
conda create -n memgen python=3.10
conda activate memgen
pip install -r requirements.txt

cp scripts/experiments/server.env.example scripts/experiments/.server.env
```

编辑 `scripts/experiments/.server.env`，填写服务器上的输出根目录、GPU 编号和所需
checkpoint 的绝对路径。该文件已被 Git 忽略，不能提交。

## 运行实验

每次实验先同步 `main`，再运行对应的一键脚本：

```bash
git pull --ff-only origin main
bash scripts/experiments/<dataset>/run_<method>.sh
```

脚本会自行指定描述实验核心的 `run-id`，因此输出目录包含数据集、方法、关键超参、阶段
与时间戳。每次新的训练或评测必须同时新增：

1. `configs/experiments/<dataset>/<method>.yaml`
2. `scripts/experiments/<dataset>/run_<method>.sh`

## 当前：GSM8K entropy gate

该实验使用最后一层、去除前 4 个 attention-sink token 后的注意力熵，在 delimiter
边界决定是否调用 Weaver。先在 validation 集校准阈值：

```bash
bash scripts/experiments/gsm8k/run_entropy_calibration_sink4_q85.sh
```

脚本结束后会打印生成的 `entropy_threshold.json` 路径。把它传给 test 脚本：

```bash
bash scripts/experiments/gsm8k/run_entropy_eval_sink4_q85.sh \
  /absolute/path/to/entropy_threshold.json
```

每次评测的 `evaluate/` 目录包含：

- `answer.json`：模型回答与结果；
- `augmentation_positions.csv`：实际 latent 插入位置；
- `entropy_gate_trace.csv`：每个 delimiter 候选点的 entropy 与门控决定。
