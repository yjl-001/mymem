# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

MemGen (ICLR 2026) is a framework that empowers LLM agents to evolve through experience by generating latent memory tokens directly within the model's reasoning stream — no parameter updates or external databases. It has two core modules:

- **Memory Weaver** — Synthesizes past context into compact latent token sequences, inserted at delimiter boundaries (`,`, `.`, `\n`) during reasoning. Separate latent parameters exist for prompt augmentation (before generation starts) and inference augmentation (during autoregressive generation).
- **Memory Trigger** — A binary classifier that decides *when* to insert latent memory at each delimiter during generation. Trained via GRPO.

Training is two-stage: first train the Weaver (SFT or GRPO), then optionally train the Trigger (GRPO only). The Weaver and Trigger are separate LoRA-tuned copies of the base model, with the base *reasoner* kept frozen.

## Architecture

```
MemGenModel (PreTrainedModel)
├── reasoner          — Frozen base LLM (e.g., Qwen2.5-1.5B-Instruct)
├── weaver            — LoRA-tuned model that generates latent memory tokens
│   ├── prompt_query_latents      — Learnable query latents for prompt augmentation
│   ├── inference_query_latents   — Learnable query latents for inference augmentation
│   ├── prompt_latent_ln/scale    — Per-type normalization
│   └── inference_latent_ln/scale
├── trigger           — LoRA-tuned binary classifier for augmentation gating
│   └── output_layer  — Linear(hidden_size, 2) predicting [no_augment, augment]
├── reasoner_to_weaver — Linear projection: reasoner embeddings → weaver hidden space
└── weaver_to_reasoner — Linear projection: weaver hidden states → reasoner embedding space
```

**Data flow during training (`_forward`):**
1. Detect augmentation points: prompt-label boundary (prompt aug) and post-delimiter positions in labels (inference aug)
2. Iterate through segments: slice embeddings → project to weaver space → run weaver's `augment_prompt`/`augment_infer` → project back → concatenate latent embeddings into sequence
3. Feed augmented embedding sequence to frozen reasoner → compute loss only on valid (non-latent) positions

**Data flow during generation (`generate`):**
1. At each step, the Trigger decides whether to insert latent memory at the current position
2. If augmenting: project current context to weaver, run augmentation, project back, insert latent tokens
3. Non-augmenting sequences get left-padded to maintain uniform batch dimensions
4. Continue autoregressive generation with the frozen reasoner

**Two forward modes:**
- `INSTRUCTION_STATE` (`_instructional_forward`) — Single-turn: exactly one prompt augmentation point at the prompt→completion boundary
- `CONVERSATION_STATE` (`_conversational_forward`) — Multi-turn: detects `<|im_start|>assistant\n` boundaries, runs `_forward` per turn. Batch size must be 1.

## Key Files

| File | Role |
|------|------|
| `main.py` | Entry point. Parses args, builds config, instantiates model + data + runner |
| `memgen/runner.py` | `MemGenRunner` — orchestrates training/evaluation, creates trainers, manages datasets |
| `memgen/model/modeling_memgen.py` | `MemGenModel` — core model with `forward`, `generate`, save/load |
| `memgen/model/modeling_utils.py` | Mixins: `MemGenLoraSwitchMixin` (LoRA insert/fix/open), `MemGenGenerationMixin` (generation helpers) |
| `memgen/model/weaver.py` | `MemGenWeaver` — latent augmentation module |
| `memgen/model/trigger.py` | `MemGenTrigger` — binary augmentation classifier |
| `memgen/model/configuration_memgen.py` | `MemGenConfig` (extends `PretrainedConfig`) |
| `memgen/trainer/weaver_grpo_trainer.py` | `WeaverGRPOTrainer` — extends `trl.GRPOTrainer` for weaver GRPO |
| `memgen/trainer/trigger_grpo_trainer.py` | `TriggerGRPOTrainer` — extends `trl.GRPOTrainer` for trigger GRPO |
| `memgen/utils.py` | Chat template, eval recorders, param utilities, `gather_objects` |
| `common/config.py` | `Config` — OmegaConf-based config builder with CLI override support |
| `data/__init__.py` | Dataset builder registry (`_DATA_BUILDER_MAP`) |
| `data/base_builder.py` | `BaseBuilder` — abstract dataset builder (SFT/RL modes) |
| `data/base_env.py` | `BaseEnv` / `StaticEnv` / `DynamicEnv` — environment abstraction |
| `interactions/base_interaction.py` | `InteractionManager` / `InteractionDataProto` — generation loop abstraction |
| `interactions/singleturn_interaction.py` | `SingleTurnInteractionManager` — for StaticEnv datasets |
| `interactions/multiturn_interaction.py` | `MultiTurnInteractionManager` — for DynamicEnv datasets (multi-step agent loops) |
| `configs/latent_memory/*.yaml` | Per-dataset configs (kodcode, gsm8k, gpqa, triviaqa) |
| `configs/zero2.yaml` | DeepSpeed ZeRO-2 config for Accelerate |

## Commands

### Training

All training uses `accelerate launch` with DeepSpeed ZeRO-2. Only DDP is supported (no FSDP).

**Weaver SFT:**
```bash
bash scripts/weaver_sft.sh
```
Or directly:
```bash
python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes=$NUM_GPUS \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${REASONER_MODEL} \
    model.weaver.model_name ${WEAVER_MODEL} \
    datasets.mode sft \
    run.mode train \
    run.train_weaver True \
    run.train_trigger False \
    run.train_weaver_method sft
```

**Weaver GRPO:**
```bash
bash scripts/weaver_grpo.sh
```

**Trigger GRPO** (requires a pre-trained Weaver checkpoint):
```bash
bash scripts/trigger_train.sh
```

Key config overrides via `--options`: `model.*`, `run.*`, `datasets.mode`. The `--options` use dot-separated paths mapping to the YAML structure (OmegaConf dotlist format).

### Evaluation

```bash
bash scripts/eval.sh
```
Or use dataset-specific eval scripts in `scripts/eval/`. Set `LOAD_MODEL_PATH` in the script to your checkpoint directory before running.

To evaluate a vanilla (non-MemGen) model: replace the `generate` function in `memgen/model/modeling_memgen.py` with the commented-out version (lines 379–450).

### Environment Setup

```bash
conda create -n memgen python=3.10
conda activate memgen
pip install -r requirements.txt
```

## Configuration System

- Base configs: `configs/latent_memory/{dataset_name}.yaml` — defines model, dataset, run, and interaction settings
- CLI overrides: `--options key.path value` — merged via OmegaConf on top of the base YAML
- Config is split into three sections consumed separately: `model`, `dataset`, `run`
- Working directories auto-generated under `.cache/{mode}/{dataset}/{model}/pn={x}_pl={y}_in={z}_il={w}_{timestamp}/`

## Supported Models & Datasets

**Base models:** Qwen2.5-1.5B-Instruct, SmolLM3-3B
**Datasets:** GSM8K (math), KodCode (code), GPQA (science QA), TriviaQA (knowledge)

Each dataset has a `Builder` (data loading/splitting) and an `Env` (reward computation, environment interaction).

- StaticEnv datasets (GSM8K, KodCode, GPQA): single-turn, reward computed by comparing answer to ground truth
- DynamicEnv datasets (TriviaQA): multi-turn with search/retrieval interaction

### Dataset-specific augmentation params:
| Dataset | `max_prompt_aug_num` | `max_inference_aug_num` |
|---------|---------------------|------------------------|
| GSM8K, GPQA, KodCode | 1 | 5 |
| TriviaQA | 6 | 0 |

## Critical Implementation Details

- **Chat template:** The codebase forces `CONVERSATION_TEMPLATE` (ChatML/Jinja from SmolLM3) via `tokenizer.chat_template = CONVERSATION_TEMPLATE`. This is essential — using the wrong template will break label masking and conversation detection.
- **Formatting sensitivity:** For the 1.5B model, appending `\n` vs `.` after `\boxed{}` in the prompt significantly affects performance. The current codebase uses the ChatML template consistently.
- **DDP only:** FSDP is not supported. Use `configs/zero2.yaml` with DeepSpeed ZeRO-2.
- **bfloat16:** Models are loaded and evaluated in `torch.bfloat16`. Training configs set `bf16: True`.
- **Label masking:** The `_postprocess_assistant_labels` method masks `<|im_start|>assistant\n` tokens in labels to -100 so they don't contribute to loss.
- **Latent tokens:** Injected between real tokens at delimiter boundaries. A `current_latents_mask` tracks which positions are latent (excluded from loss computation).
- **Padding strategy:** Left-padding by default (`tokenizer.padding_side = "left"`), with dynamic left-padding during batched generation to keep non-augmented sequences aligned.
- **Save format:** `save_pretrained` writes `projs.bin` (projection layers), `weaver.bin` (latent params), `trigger.bin` (classifier head), plus LoRA adapter weights in `weaver/` and `trigger/` subdirectories.
- **Trigger training:** The trigger is a separate model copy with its own LoRA adapters. Its output layer is a binary classifier trained to predict whether to insert latent memory at each delimiter position. When `trigger.active=False`, it defaults to always predicting "augment" (logit for class 1 = 1.0).
