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

## 2. 服务器检出指定代码版本

服务器保留一个只用于 `fetch` 的主 clone，然后为每个提交创建一个独立 worktree：

```bash
cd /path/to/MemGen-mirror
bash scripts/create_server_worktree.sh /mnt/worktrees/memgen origin/feature/mi-entropy-gate
cd /mnt/worktrees/memgen/memgen-<timestamp>-<sha>
```

这样后续 `git fetch` 不会改变正在训练的源码。若服务器空间有限，也可以在确认没有运行
任务、且工作区干净后执行 `git pull --ff-only`；不要使用会丢失未提交内容的同步命令。

首次在服务器使用时，在 worktree 中创建环境并缓存依赖/模型。建议将两者放在仓库外：

```bash
conda create -n memgen python=3.10
conda activate memgen
pip install -r requirements.txt
export HF_HOME=/mnt/cache/huggingface
export MEMGEN_OUTPUT_ROOT=/mnt/experiments/memgen-runs
```

`MEMGEN_OUTPUT_ROOT` 未设置时仍兼容旧行为，结果写入当前仓库的 `.cache/`。

## 3. 参数文件与覆盖

`configs/latent_memory/*.yaml` 是各数据集的基线。`configs/experiments/**/*.yaml`
只保存一次实验相对基线的差异，并指定 `base_cfg_path` 与 GPU 启动参数。新的正式实验
应复制最接近的文件并提交，例如：

```bash
cp configs/experiments/kodcode/weaver_sft.yaml \
   configs/experiments/kodcode/mi_entropy_weaver_sft.yaml
```

需要临时改变一个值时，使用 `--set key=value`；这不会污染已提交的 YAML：

```bash
python scripts/launch_experiment.py train \
  configs/experiments/kodcode/weaver_sft.yaml \
  --set run.weaver.sft.learning_rate=2e-5 \
  --set model.weaver.prompt_latents_len=16
```

## 4. 训练、评测与 checkpoint 衔接

训练：

```bash
python scripts/launch_experiment.py train \
  configs/experiments/kodcode/weaver_sft.yaml
```

评测始终显式传入 checkpoint 的**绝对路径**，不根据时间倒序猜测“最新模型”：

```bash
python scripts/launch_experiment.py eval \
  configs/experiments/kodcode/eval.yaml \
  --set model.load_model_path=/mnt/experiments/memgen-runs/train/kodcode/Qwen2.5-1.5B-Instruct/<run>/model
```

每一次运行会在输出目录写入：

- `resolved_config.yaml`：合并基线 YAML、实验 YAML、`--set` 后的完整参数；
- `run_manifest.json`：Git commit、分支、是否 dirty、启动参数、机器名和时间；
- 原有的 `model/`、`evaluate/answer.json`、日志与 TensorBoard 文件。

正式报告结果时，至少记录：Git commit、实验 YAML 路径、完整 checkpoint 路径、
`resolved_config.yaml` 路径、随机种子和评测输出路径。

## 5. 建议的阶段配置

不要用一个巨大的 shell pipeline 隐式串联所有阶段。每个阶段是一份独立、可提交的配置：

1. `weaver_sft.yaml`：从基础模型训练 Weaver；
2. `weaver_grpo.yaml`：通过 `model.load_model_path` 明确指向 SFT checkpoint；
3. `trigger_grpo.yaml`：明确指向 Weaver checkpoint；
4. `eval.yaml`：明确指向待测 checkpoint。

这使得任一阶段能够单独重跑，也让消融实验只改变必要的一个配置字段。
