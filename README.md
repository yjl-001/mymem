# MemGen

MemGen 让 Agent 在推理流中插入 latent memory，而不更新 reasoner 参数。当前仓库同时保留
原始 MemGen 训练框架，以及 verifier-backed experience memory 研究主线。模型、数据集、
checkpoint、日志和实验产物只保存在服务器，不进入 Git。

当前经验记忆结论与系统契约见
[experience_calibrated_steering_plan.md](docs/codex/experience_calibrated_steering_plan.md)，
可运行入口见 [scripts/experiments/README.md](scripts/experiments/README.md)。

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

## 当前经验记忆评测

完整系统只评测三种条件：`vanilla`、`gate_observation_only` 和
`matched_persistent_memory`。服务器同步 `main` 后按顺序运行 Phase 1、risk gate、E0 与 E1D；
命令和所需 artifact 路径见实验入口文档。旧过程性实验与负向路线可从 Git 历史恢复。
