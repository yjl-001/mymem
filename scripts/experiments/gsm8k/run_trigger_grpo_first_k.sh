#!/usr/bin/env bash
# Train the GSM8K Trigger with GRPO from a completed first-k Weaver checkpoint.
# Edit the values in this block, then run this file directly.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# ===== Edit this block for a new Trigger-GRPO run =====
CUDA_VISIBLE_DEVICES="7"
MAIN_PROCESS_PORT=29509
OUTPUT_ROOT="$REPO_ROOT/.cache"
LOG_PATH="$REPO_ROOT/gsm8k_trigger_grpo_first_k.log"

# Set this to the `model/` directory produced by the Weaver SFT stage.
LOAD_MODEL_PATH="/absolute/path/to/gsm8k_weaver_sft_first_k_run/model"

# These architecture settings must match the Weaver checkpoint exactly.
REASONER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
WEAVER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
TRIGGER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=3
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8
WEAVER_INSERTION_STRATEGY="first_k"

# Trigger GRPO rollout/training settings.
TRIGGER_BATCH_SIZE=8
NUM_GENERATIONS=8
GRADIENT_ACCUMULATION_STEPS=1
NUM_TRAIN_EPOCHS=1
LEARNING_RATE=1e-5
MAX_PROMPT_LENGTH=1024
MAX_COMPLETION_LENGTH=1024
TEMPERATURE=1.0

NCCL_DEBUG=INFO
NCCL_IB_DISABLE=1
NCCL_P2P_DISABLE=1
NCCL_ASYNC_DISABLE=1
# ===== End editable block =====

if [[ ! -d "$LOAD_MODEL_PATH" ]]; then
  echo "Weaver checkpoint directory not found: $LOAD_MODEL_PATH" >&2
  echo "Set LOAD_MODEL_PATH to the Weaver SFT output's model/ directory." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES
export MEMGEN_OUTPUT_ROOT="$OUTPUT_ROOT"
export NCCL_DEBUG NCCL_IB_DISABLE NCCL_P2P_DISABLE NCCL_ASYNC_DISABLE

NUM_GPUS="$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | wc -l | tr -d ' ')"
echo "Using $NUM_GPUS GPU(s): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

RUN_TAG="$(date +%Y%m%d-%H%M%S)"
RUN_ID="gsm8k_trigger_grpo_first_k_${RUN_TAG}"

python scripts/launch_experiment.py train \
  configs/experiments/gsm8k/trigger_grpo.yaml \
  --devices "$CUDA_VISIBLE_DEVICES" \
  --num-processes "$NUM_GPUS" \
  --main-process-port "$MAIN_PROCESS_PORT" \
  --run-id "$RUN_ID" \
  --set model.model_name="$REASONER_MODEL" \
  --set model.load_model_path="$LOAD_MODEL_PATH" \
  --set model.max_prompt_aug_num="$MAX_PROMPT_AUG_NUM" \
  --set model.max_inference_aug_num="$MAX_INFERENCE_AUG_NUM" \
  --set model.weaver.model_name="$WEAVER_MODEL" \
  --set model.weaver.prompt_latents_len="$PROMPT_LATENTS_LEN" \
  --set model.weaver.inference_latents_len="$INFERENCE_LATENTS_LEN" \
  --set model.weaver.insertion_strategy.name="$WEAVER_INSERTION_STRATEGY" \
  --set model.trigger.model_name="$TRIGGER_MODEL" \
  --set model.trigger.active=true \
  --set dataset.mode=grpo \
  --set run.mode=train \
  --set run.train_weaver=false \
  --set run.train_trigger=true \
  --set run.train_trigger_method=grpo \
  --set run.trigger.grpo.per_device_train_batch_size="$TRIGGER_BATCH_SIZE" \
  --set run.trigger.grpo.per_device_eval_batch_size="$TRIGGER_BATCH_SIZE" \
  --set run.trigger.grpo.num_generations="$NUM_GENERATIONS" \
  --set run.trigger.grpo.gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
  --set run.trigger.grpo.num_train_epochs="$NUM_TRAIN_EPOCHS" \
  --set run.trigger.grpo.learning_rate="$LEARNING_RATE" \
  --set run.trigger.grpo.max_prompt_length="$MAX_PROMPT_LENGTH" \
  --set run.trigger.grpo.max_completion_length="$MAX_COMPLETION_LENGTH" \
  --set run.trigger.grpo.temperature="$TEMPERATURE" \
  2>&1 | tee "$LOG_PATH"
