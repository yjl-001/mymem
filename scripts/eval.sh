#!/bin/bash

export DEBUG_MODE=true  
export LOG_PATH="./debug_log_2b.txt"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MAIN_PROCESS_PORT=29508

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Using $NUM_GPUS GPU(s): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1

# options:
# - Qwen/Qwen2.5-1.5B-Instruct
# - HuggingFaceTB/SmolLM3-3B
REASONER_MODEL=${REASONER_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
WEAVER_MODEL=${WEAVER_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
TRIGGER_MODEL=${TRIGGER_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
TRIGGER_ACTIVE=${TRIGGER_ACTIVE:-False}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}


# Dataset configs
DATASET_NAME=${DATASET_NAME:-kodcode}  # gsm8k, gpqa, kodcode, triviaqa

# MemGen configs

# Augmentation configs:
# - For gsm8k, gpqa, kodcode: MAX_PROMPT_AUG_NUM=1, MAX_INFERENCE_AUG_NUM=5
# - For triviaqa:             MAX_PROMPT_AUG_NUM=8, MAX_INFERENCE_AUG_NUM=0
MAX_PROMPT_AUG_NUM=${MAX_PROMPT_AUG_NUM:-1}
MAX_INFERENCE_AUG_NUM=${MAX_INFERENCE_AUG_NUM:-0}
PROMPT_LATENTS_LEN=${PROMPT_LATENTS_LEN:-16}
INFERENCE_LATENTS_LEN=${INFERENCE_LATENTS_LEN:-16}
WEAVER_INSERTION_STRATEGY=${WEAVER_INSERTION_STRATEGY:-first_k}
WEAVER_SINK_SCORE_THRESHOLD=${WEAVER_SINK_SCORE_THRESHOLD:-0.3}
WEAVER_SINK_SCORE_LAYER_WINDOW=${WEAVER_SINK_SCORE_LAYER_WINDOW:-4}
SINK_AWARE_GENERATION=${SINK_AWARE_GENERATION:-False}

BATCH_SIZE=${BATCH_SIZE:-4}
case "${SINK_AWARE_GENERATION}" in
    True|true|1)
        SINK_AWARE_GENERATION=True
        ;;
    False|false|0)
        SINK_AWARE_GENERATION=False
        ;;
    *)
        echo "SINK_AWARE_GENERATION must be one of: True, False, true, false, 1, 0" >&2
        exit 1
        ;;
esac

if [ "${SINK_AWARE_GENERATION}" = "True" ]; then
    if [ "${BATCH_SIZE}" -ne 1 ]; then
        echo "Sink-aware generation requires BATCH_SIZE=1" >&2
        exit 1
    fi
    if [ "${WEAVER_INSERTION_STRATEGY}" = "first_k" ]; then
        echo "Sink-aware generation requires a sink threshold strategy" >&2
        exit 1
    fi
    if [ "${MAX_INFERENCE_AUG_NUM}" -le 0 ]; then
        echo "Sink-aware generation requires MAX_INFERENCE_AUG_NUM>0" >&2
        exit 1
    fi
fi

# Trained model path: 
# - Must point to a MemGen checkpoint directory containing config.json,
#   projs.bin, weaver.bin, trigger.bin, weaver/, and trigger/.
# - Required when evaluating the model
LOAD_MODEL_PATH=${LOAD_MODEL_PATH:-""}

# evaluate
python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes=${NUM_GPUS} \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${REASONER_MODEL} \
    model.attn_implementation ${ATTN_IMPLEMENTATION} \
    model.load_model_path ${LOAD_MODEL_PATH} \
    model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
    model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
    model.weaver.model_name ${WEAVER_MODEL} \
    model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
    model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
    model.weaver.insertion_strategy.name ${WEAVER_INSERTION_STRATEGY} \
    model.weaver.insertion_strategy.sink_score_threshold ${WEAVER_SINK_SCORE_THRESHOLD} \
    model.weaver.insertion_strategy.sink_score_layer_window ${WEAVER_SINK_SCORE_LAYER_WINDOW} \
    model.trigger.model_name ${TRIGGER_MODEL} \
    model.trigger.active ${TRIGGER_ACTIVE} \
    run.mode evaluate \
    run.interaction.batch_size ${BATCH_SIZE} \
    run.interaction.sink_aware_generation ${SINK_AWARE_GENERATION} \
    run.interaction.temperature 0.0 \
    run.interaction.max_response_length 1024 \
