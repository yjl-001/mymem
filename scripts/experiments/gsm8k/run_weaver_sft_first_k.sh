#!/usr/bin/env bash
# Train the GSM8K Weaver with first-k delimiter insertion.
# Edit the values in this block, then run this file directly. No shell exports
# or .server.env file are required.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# ===== Edit this block for a new run =====
DEBUG_MODE=true
LOG_PATH="$REPO_ROOT/gsm8k_weaver_sft_first_k.log"
CUDA_VISIBLE_DEVICES="7"
MAIN_PROCESS_PORT=29507
OUTPUT_ROOT="$REPO_ROOT/.cache"

REASONER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
WEAVER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
TRIGGER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
LOAD_MODEL_PATH=null

MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=3
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8
WEAVER_INSERTION_STRATEGY="first_k"
WEAVER_SINK_SCORE_THRESHOLD=0.7
WEAVER_SINK_SCORE_LAYER_WINDOW=1
SINK_AWARE_GENERATION=false

BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=1
NUM_TRAIN_EPOCHS=2

NCCL_DEBUG=INFO
NCCL_IB_DISABLE=1
NCCL_P2P_DISABLE=1
NCCL_ASYNC_DISABLE=1
# ===== End editable block =====

export DEBUG_MODE LOG_PATH CUDA_VISIBLE_DEVICES MAIN_PROCESS_PORT
export MEMGEN_OUTPUT_ROOT="$OUTPUT_ROOT"
export NCCL_DEBUG NCCL_IB_DISABLE NCCL_P2P_DISABLE NCCL_ASYNC_DISABLE

NUM_GPUS="$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | wc -l | tr -d ' ')"
echo "Using $NUM_GPUS GPU(s): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

RUN_TAG="$(date +%Y%m%d-%H%M%S)"
RUN_ID="gsm8k_weaver_sft_first_k_${RUN_TAG}"

python scripts/launch_experiment.py train \
  configs/experiments/gsm8k/weaver_sft.yaml \
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
  --set model.weaver.insertion_strategy.sink_score_threshold="$WEAVER_SINK_SCORE_THRESHOLD" \
  --set model.weaver.insertion_strategy.sink_score_layer_window="$WEAVER_SINK_SCORE_LAYER_WINDOW" \
  --set model.trigger.model_name="$TRIGGER_MODEL" \
  --set model.trigger.active=false \
  --set dataset.mode=sft \
  --set run.mode=train \
  --set run.train_weaver=true \
  --set run.train_weaver_method=sft \
  --set run.train_trigger=false \
  --set run.weaver.sft.per_device_train_batch_size="$BATCH_SIZE" \
  --set run.weaver.sft.per_device_eval_batch_size="$BATCH_SIZE" \
  --set run.weaver.sft.bf16=true \
  --set run.weaver.sft.gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
  --set run.weaver.sft.num_train_epochs="$NUM_TRAIN_EPOCHS" \
  --set run.interaction.sink_aware_generation="$SINK_AWARE_GENERATION" \
  2>&1 | tee "$LOG_PATH"
