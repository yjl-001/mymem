# MemGen: Weaving Generative Latent Memory for Self-Evolving Agents


## 👋 Introduction
This repo is the official implementation of [**[ICLR 2026] MemGen: Weaving Generative Latent Memory for Self-Evolving Agents**](https://arxiv.org/pdf/2509.24704).

Inspired by the human brain’s ability to dynamically integrate memory and reasoning, MemGen introduces a novel framework that empowers AI agents to evolve through experience—without relying on rigid parameter updates or external databases.

Unlike traditional approaches, MemGen generates latent memory tokens directly within the model’s reasoning stream. It features:
- A Memory Trigger that decides when to recall memory.
- A Memory Weaver that synthesizes past experiences into compact, latent sequences—seamlessly enriching ongoing reasoning.

![alt text](assets/memgen.png)


## ❓ FAQ

#### Q1: Why does the code encounter issues when running on multiple GPUs?

**A:** DDP is supported, but FSDP is not currently supported. Thank you for your understanding.


#### Q2: Where is the multi-turn GRPO code (e.g., for AlfWorld and TriviaQA)?

**A:** We plan to release the MemGen-GRPO eval/train scripts and checkpoints after releasing those for MemGen-SFT. Thank you for your patience and understanding.


#### Q3: What improvements are included in the latest MemGen codebase?

**A:** In the previous version, single-turn training did not use the ChatML template (for both the baseline and MemGen), which led to lower performance. In addition, we identified a small but impactful formatting issue: for the 1.5B model, whether the prompt ends with `\boxed{}` followed by `.` or `\n` significantly affects performance. In particular, appending  `\n` after *“Put your answer within \boxed{}”* can noticeably degrade results compared with appending `.`. While surprising, this behavior was consistent in our tests. The updated codebase consistently applies the ChatML template across all datasets and resolves these formatting inconsistencies. We still observe stable performance gains from MemGen under this unified setup.

We apologize for any inconvenience caused by the earlier version.

## 🌎 Setup

Create and activate the MemGen environment:  
Option 1: Install via `requirements.txt`
```
conda create -n memgen python=3.10
conda activate memgen
pip install -r requirements.txt
```

Option 2: Install via `memgen.yml`
```
conda env create -f memgen.yml
conda activate memgen
```

Option 3: Set Up Search Environment  
Please follow the instructions in the [Search-R1](https://github.com/PeterGriffinJin/Search-R1?tab=readme-ov-file#retriever-environment-optional) to configure the retriever environment.

## 🤗 Quick Evaluation

All training and evaluation use the same versioned launcher. Select an
experiment YAML, then supply the server GPU and the exact MemGen checkpoint:

```bash
python scripts/launch_experiment.py eval configs/experiments/kodcode/eval.yaml \
  --devices 0 \
  --set model.load_model_path=/data/memgen-runs/train/<run>/model
```

The launcher rejects an empty or incomplete checkpoint, and records the
resolved configuration and Git commit beside the evaluation result.


## ▶️ How to Run
MemGen consists of **two modules**: *Weaver* and *Trigger*. We follow a two-stage training approach, training each module separately.

### Versioned local-to-server experiments

For development on a local machine and reproducible server-side training, see
[the experiment workflow](docs/experiment_workflow.md). It provides a unified
launcher based on versioned YAML experiment files, explicit checkpoint paths,
and automatic snapshots of the resolved parameters and Git commit.

The canonical KodCode templates are `weaver_sft.yaml`, `weaver_grpo.yaml`,
`trigger_grpo.yaml`, and `eval.yaml` under `configs/experiments/kodcode/`.
For the full local-to-server workflow, see
[docs/experiment_workflow.md](docs/experiment_workflow.md).



## 🫡 Citation
If you find this repository helpful, a citation to our paper would be greatly appreciated:
```
@article{zhang2025memgen,
  title={MemGen: Weaving Generative Latent Memory for Self-Evolving Agents},
  author={Zhang, Guibin and Fu, Muxin and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2509.24704},
  year={2025}
}
```

## 🙏 Acknowledgement
- We sincerely thank [Search-R1](https://github.com/PeterGriffinJin/Search-R1) for open-sourcing their search web environment.
- We sincerely thank the previous latent reasoning works such as [LatentSeek](https://arxiv.org/abs/2505.13308), [SoftCoT](https://arxiv.org/abs/2502.12134), [R3Mem](https://arxiv.org/abs/2502.15957v1) and so on.
- We also extend our heartfelt thanks to [LAVIS](https://github.com/salesforce/LAVIS) for their code framework design.
