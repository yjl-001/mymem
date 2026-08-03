# MemGen 本地开发与服务器实验流程

这套流程将 **代码版本**、**实验参数** 与 **运行产物** 分开管理：GitHub
保存前两者；数据集、Hugging Face 缓存、checkpoint、日志与评测结果保存在服务器的
磁盘中，不进入 Git。

## 1. 本地开发和同步

每一项可运行的方法改动都应有一个分支和一个提交。实验配置也随代码一起提交。

```bash
git switch -c feature/mi-entropy-gate
git add <changed-source-files> configs/experiments/
git commit -m "feat: add MI entropy-gated memory bank"
git push -u origin feature/mi-entropy-gate
```

不要在服务器修改、提交源码；服务器仅运行已推送的提交。模型权重、数据和 `.cache/`
均由 `.gitignore` 排除。若本地工作区有尚未提交的改动，运行清单会标为 `git_dirty: true`，
但这类运行不应作为正式结果。

## 2. 服务器同步代码

本地与服务器共同使用 `main` 分支。服务器不修改源码，只在开始新任务前同步：

```bash
cd /path/to/MemGen
git switch main
git pull --ff-only origin main
```

首次在服务器使用时，创建环境并缓存依赖/模型。建议将两者放在仓库外：

```bash
conda create -n memgen python=3.10
conda activate memgen
pip install -r requirements.txt
export HF_HOME=/mnt/cache/huggingface
export MEMGEN_OUTPUT_ROOT=/mnt/experiments/memgen-runs
```

`MEMGEN_OUTPUT_ROOT` 未设置时仍兼容旧行为，结果写入当前仓库的 `.cache/`。

## 3. 参数文件、一键脚本与覆盖

`configs/latent_memory/*.yaml` 是各数据集的基线。`configs/experiments/**/*.yaml`
只保存一次实验相对基线的差异，并指定 `base_cfg_path` 与 GPU 启动参数。每个正式训练或
评测还必须有一个匹配的 `scripts/experiments/**/*.sh` 一键脚本；二者都随代码提交。脚本
读取服务器本地 `.server.env` 中的路径与设备配置，并固定描述实验核心的 `run-id`。

新的正式实验应复制最接近的配置和脚本，例如：

```bash
cp configs/experiments/kodcode/weaver_sft.yaml \
   configs/experiments/kodcode/mi_entropy_weaver_sft.yaml
cp scripts/experiments/gsm8k/run_entropy_calibration_sink4_q85.sh \
   scripts/experiments/kodcode/run_mi_entropy_weaver_sft.sh
```

需要临时改变一个值时，使用 `--set key=value`；这不会污染已提交的 YAML：

临时覆盖参数应写进本次实验对应的 Bash 脚本；脚本与配置共同构成完整、可重跑的实验定义。
不要把正式实验参数仅留在聊天记录或服务器命令历史中。

## 4. 训练、评测与 checkpoint 衔接

训练或评测时，服务器只执行版本化的一键脚本：

```bash
git pull --ff-only origin main
bash scripts/experiments/<dataset>/run_<method>.sh
```

需要前一阶段 checkpoint 的脚本应读取 `.server.env` 或接受其**绝对路径**作为参数，
不根据时间倒序猜测“最新模型”。

```bash
python scripts/launch_experiment.py eval \
  configs/experiments/kodcode/eval.yaml \
  --devices 0 \
  --set model.load_model_path=/mnt/experiments/memgen-runs/train/kodcode/Qwen2.5-1.5B-Instruct/<run>/model
```

每一次运行会在输出目录写入：

- `resolved_config.yaml`：合并基线 YAML、实验 YAML、`--set` 后的完整参数；
- `run_manifest.json`：Git commit、分支、是否 dirty、启动参数、机器名和时间；
- 原有的 `model/`、`evaluate/answer.json`、日志与 TensorBoard 文件。

正式报告结果时，至少记录：Git commit、实验 YAML 路径、完整 checkpoint 路径、
`resolved_config.yaml` 路径、随机种子和评测输出路径。

## 5. 建议的阶段配置

不要使用旧的 `scripts/train/*.sh`、`scripts/eval/*.sh` 或 `scripts/pipeline.sh`。
每个阶段是一份独立、可提交的配置：

1. `weaver_sft.yaml`：从基础模型训练 Weaver；
2. `weaver_grpo.yaml`：通过 `model.load_model_path` 明确指向 SFT checkpoint；
3. `trigger_grpo.yaml`：明确指向 Weaver checkpoint；
4. `eval.yaml`：明确指向待测 checkpoint。

这使得任一阶段能够单独重跑，也让消融实验只改变必要的一个配置字段。
