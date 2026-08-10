#!/usr/bin/env bash
# Evaluate a GSM8K Weaver-SFT checkpoint with deterministic first-k insertion.
# Edit the values in this block, then run this file directly.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# ===== Edit this block for a new evaluation =====
CUDA_VISIBLE_DEVICES="7"
MAIN_PROCESS_PORT=29508
OUTPUT_ROOT="$REPO_ROOT/.cache"
LOG_PATH="$REPO_ROOT/gsm8k_weaver_sft_first_k_eval.log"

# Set this to the `model/` directory created by run_weaver_sft_first_k.sh.
LOAD_MODEL_PATH="/absolute/path/to/gsm8k_weaver_sft_first_k_run/model"

# These architecture settings must match the checkpoint being evaluated.
REASONER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
WEAVER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
TRIGGER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=3
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8
WEAVER_INSERTION_STRATEGY="first_k"

EVAL_BATCH_SIZE=8
MAX_RESPONSE_LENGTH=1024
TEMPERATURE=0.0
EVALUATION_SPLIT="test"

NCCL_DEBUG=INFO
NCCL_IB_DISABLE=1
NCCL_P2P_DISABLE=1
NCCL_ASYNC_DISABLE=1
# ===== End editable block =====

if [[ ! -d "$LOAD_MODEL_PATH" ]]; then
  echo "Checkpoint directory not found: $LOAD_MODEL_PATH" >&2
  echo "Set LOAD_MODEL_PATH to the training output's model/ directory." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES
export MEMGEN_OUTPUT_ROOT="$OUTPUT_ROOT"
export NCCL_DEBUG NCCL_IB_DISABLE NCCL_P2P_DISABLE NCCL_ASYNC_DISABLE

NUM_GPUS="$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | wc -l | tr -d ' ')"
echo "Using $NUM_GPUS GPU(s): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

RUN_TAG="$(date +%Y%m%d-%H%M%S)"
RUN_ID="gsm8k_weaver_sft_first_k_eval_${RUN_TAG}"

python scripts/launch_experiment.py eval \
  configs/experiments/gsm8k/weaver_sft_first_k_eval.yaml \
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
  --set model.trigger.active=false \
  --set dataset.mode=sft \
  --set run.mode=evaluate \
  --set run.train_weaver=false \
  --set run.train_trigger=false \
  --set run.interaction.evaluation_split="$EVALUATION_SPLIT" \
  --set run.interaction.batch_size="$EVAL_BATCH_SIZE" \
  --set run.interaction.max_response_length="$MAX_RESPONSE_LENGTH" \
  --set run.interaction.temperature="$TEMPERATURE" \
  --set run.interaction.weaver_do_sample=false \
  --set run.interaction.trigger_do_sample=false \
  --set run.interaction.sink_aware_generation=false \
  2>&1 | tee "$LOG_PATH"
