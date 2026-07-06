#!/bin/bash
set -e

# ============================================================================
# MemGen 全流水线脚本：Weaver SFT → Weaver GRPO → Trigger GRPO → 评测
#
# 用法:
#   bash scripts/pipeline.sh
#
# 可选环境变量:
#   DATASET_NAME     - 数据集 (默认 kodcode, 可选 gsm8k/gpqa/kodcode/triviaqa)
#   CUDA_VISIBLE_DEVICES - GPU (默认 0)
#   SKIP_SFT         - 设为 1 跳过 Weaver SFT
#   SKIP_GRPO        - 设为 1 跳过 Weaver GRPO
#   SKIP_TRIGGER     - 设为 1 跳过 Trigger GRPO
#   SKIP_EVAL        - 设为 1 跳过评测
#   LOAD_MODEL_PATH  - 从已有 checkpoint 恢复训练 (覆盖相应阶段的从零开始)
#
# 可选 batch size 覆盖 (均为 per_device 值，总有效 batch = per_device × GPU数 × grad_accum):
#   SFT_BATCH_SIZE          - Weaver SFT per_device batch (默认 1)
#   SFT_GRAD_ACCUM          - Weaver SFT 梯度累积 (默认 1)
#   GRPO_BATCH_SIZE         - Weaver GRPO per_device batch (默认 8)
#   GRPO_GRAD_ACCUM         - Weaver GRPO 梯度累积 (默认 1)
#   TRIGGER_BATCH_SIZE      - Trigger GRPO per_device batch (默认 8)
#   TRIGGER_GRAD_ACCUM      - Trigger GRPO 梯度累积 (默认 4)
#   EVAL_BATCH_SIZE         - 评测 per_device batch (默认 4)
# ============================================================================

# ===== 环境变量 =====
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "[pipeline] 使用 ${NUM_GPUS} GPU(s): CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1
export MAIN_PROCESS_PORT=29507

# ===== 模型配置 =====
REASONER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
WEAVER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
TRIGGER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"

# ===== 数据集 =====
DATASET_NAME=${DATASET_NAME:-kodcode}
echo "[pipeline] 数据集: ${DATASET_NAME}"

# ===== 增强参数 (全程一致，不可修改) =====
# 根据数据集自动设置
case ${DATASET_NAME} in
    gsm8k|gpqa|kodcode)
        MAX_PROMPT_AUG_NUM=1
        MAX_INFERENCE_AUG_NUM=5
        ;;
    triviaqa)
        MAX_PROMPT_AUG_NUM=8
        MAX_INFERENCE_AUG_NUM=0
        ;;
    *)
        echo "[pipeline] 未知数据集: ${DATASET_NAME}"
        exit 1
        ;;
esac
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

echo "[pipeline] 增强参数: PN=${MAX_PROMPT_AUG_NUM} IN=${MAX_INFERENCE_AUG_NUM} PL=${PROMPT_LATENTS_LEN} IL=${INFERENCE_LATENTS_LEN}"

# ===== Batch size 配置 =====
# 以下均为 per_device 值 (单卡)，总有效 batch = per_device × GPU数 × grad_accum
SFT_BATCH_SIZE=${SFT_BATCH_SIZE:-1}
SFT_GRAD_ACCUM=${SFT_GRAD_ACCUM:-1}
GRPO_BATCH_SIZE=${GRPO_BATCH_SIZE:-8}
GRPO_GRAD_ACCUM=${GRPO_GRAD_ACCUM:-1}
TRIGGER_BATCH_SIZE=${TRIGGER_BATCH_SIZE:-8}
TRIGGER_GRAD_ACCUM=${TRIGGER_GRAD_ACCUM:-4}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}

echo "[pipeline] Batch 配置:"
echo "  Weaver SFT:   per_device=${SFT_BATCH_SIZE} × ${NUM_GPUS}GPU × ${SFT_GRAD_ACCUM}accum = $((SFT_BATCH_SIZE * NUM_GPUS * SFT_GRAD_ACCUM))"
echo "  Weaver GRPO:  per_device=${GRPO_BATCH_SIZE} × ${NUM_GPUS}GPU × ${GRPO_GRAD_ACCUM}accum = $((GRPO_BATCH_SIZE * NUM_GPUS * GRPO_GRAD_ACCUM))"
echo "  Trigger GRPO: per_device=${TRIGGER_BATCH_SIZE} × ${NUM_GPUS}GPU × ${TRIGGER_GRAD_ACCUM}accum = $((TRIGGER_BATCH_SIZE * NUM_GPUS * TRIGGER_GRAD_ACCUM))"
echo "  Eval:         per_device=${EVAL_BATCH_SIZE} × ${NUM_GPUS}GPU = $((EVAL_BATCH_SIZE * NUM_GPUS))"

# ===== 模型短名 (用于路径匹配) =====
MODEL_SHORT=$(echo ${REASONER_MODEL} | awk -F/ '{print $2}')

# ===== 查找最近 checkpoint =====
function find_latest_checkpoint() {
    # 按时间倒序列出 pn/pl/in/il 参数匹配的所有训练目录下的 model 子目录
    local pattern=".cache/train/${DATASET_NAME}/${MODEL_SHORT}/pn=${MAX_PROMPT_AUG_NUM}_pl=${PROMPT_LATENTS_LEN}_in=${MAX_INFERENCE_AUG_NUM}_il=${INFERENCE_LATENTS_LEN}_*/model"
    local result=$(ls -td ${pattern} 2>/dev/null | head -1)
    echo "${result}"
}

# ===== 阶段开关 =====
SKIP_SFT=${SKIP_SFT:-0}
SKIP_GRPO=${SKIP_GRPO:-0}
SKIP_TRIGGER=${SKIP_TRIGGER:-0}
SKIP_EVAL=${SKIP_EVAL:-0}

# 如果提供 LOAD_MODEL_PATH，则跳过之前的阶段
if [ -n "${LOAD_MODEL_PATH}" ]; then
    echo "[pipeline] 从已有 checkpoint 恢复: ${LOAD_MODEL_PATH}"
    SKIP_SFT=1
    SKIP_GRPO=1
    LOAD_WEAVER_PATH="${LOAD_MODEL_PATH}"
fi

# ===== 记录时间戳 =====
PIPELINE_START=$(date +%s)
echo ""
echo "=============================="
echo "  MemGen Pipeline 开始"
echo "=============================="
echo ""

# ###########################################################################
#  Stage 1: Weaver SFT
# ###########################################################################
if [ "${SKIP_SFT}" = "1" ]; then
    echo "===== Stage 1: Weaver SFT (跳过) ====="
    LOAD_WEAVER_PATH="${LOAD_WEAVER_PATH:-}"
else
    echo ""
    echo "===== Stage 1: Weaver SFT ====="
    STAGE_START=$(date +%s)

    python -m accelerate.commands.launch \
        --config_file=configs/zero2.yaml \
        --num_processes=${NUM_GPUS} \
        main.py \
        --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
        --options \
        model.model_name ${REASONER_MODEL} \
        model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
        model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
        model.weaver.model_name ${WEAVER_MODEL} \
        model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
        model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
        model.trigger.model_name ${TRIGGER_MODEL} \
        model.trigger.active False \
        datasets.mode sft \
        run.mode train \
        run.train_weaver True \
        run.train_trigger False \
        run.train_weaver_method sft \
        run.weaver.sft.per_device_train_batch_size ${SFT_BATCH_SIZE} \
        run.weaver.sft.per_device_eval_batch_size ${SFT_BATCH_SIZE} \
        run.weaver.sft.bf16 True \
        run.weaver.sft.gradient_accumulation_steps ${SFT_GRAD_ACCUM}

    LOAD_WEAVER_PATH=$(find_latest_checkpoint)
    if [ -z "${LOAD_WEAVER_PATH}" ]; then
        echo "[pipeline] 错误: Stage 1 未找到 checkpoint"
        exit 1
    fi

    STAGE_END=$(date +%s)
    echo "[pipeline] Stage 1 耗时: $((STAGE_END - STAGE_START))s"
    echo "[pipeline] Weaver SFT checkpoint: ${LOAD_WEAVER_PATH}"
fi

# ###########################################################################
#  Stage 2: Weaver GRPO
# ###########################################################################
if [ "${SKIP_GRPO}" = "1" ]; then
    echo ""
    echo "===== Stage 2: Weaver GRPO (跳过) ====="
else
    echo ""
    echo "===== Stage 2: Weaver GRPO ====="
    STAGE_START=$(date +%s)

    if [ -z "${LOAD_WEAVER_PATH}" ]; then
        echo "[pipeline] 错误: 未找到 Weaver checkpoint (需要先运行 Stage 1 或设置 LOAD_MODEL_PATH)"
        exit 1
    fi
    echo "[pipeline] 加载 Weaver: ${LOAD_WEAVER_PATH}"

    python -m accelerate.commands.launch \
        --config_file=configs/zero2.yaml \
        --num_processes=${NUM_GPUS} \
        main.py \
        --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
        --options \
        model.model_name ${REASONER_MODEL} \
        model.load_model_path ${LOAD_WEAVER_PATH} \
        model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
        model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
        model.weaver.model_name ${WEAVER_MODEL} \
        model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
        model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
        model.trigger.model_name ${TRIGGER_MODEL} \
        model.trigger.active False \
        datasets.mode grpo \
        run.mode train \
        run.train_weaver True \
        run.train_trigger False \
        run.train_weaver_method grpo \
        run.weaver.grpo.per_device_train_batch_size ${GRPO_BATCH_SIZE} \
        run.weaver.grpo.per_device_eval_batch_size ${GRPO_BATCH_SIZE} \
        run.weaver.grpo.num_train_epochs 1 \
        run.weaver.grpo.num_generations 8 \
        run.weaver.grpo.gradient_accumulation_steps ${GRPO_GRAD_ACCUM}

    LOAD_WEAVER_PATH=$(find_latest_checkpoint)
    if [ -z "${LOAD_WEAVER_PATH}" ]; then
        echo "[pipeline] 错误: Stage 2 未找到 checkpoint"
        exit 1
    fi

    STAGE_END=$(date +%s)
    echo "[pipeline] Stage 2 耗时: $((STAGE_END - STAGE_START))s"
    echo "[pipeline] Weaver GRPO checkpoint: ${LOAD_WEAVER_PATH}"
fi

# ###########################################################################
#  Stage 3: Trigger GRPO
# ###########################################################################
if [ "${SKIP_TRIGGER}" = "1" ]; then
    echo ""
    echo "===== Stage 3: Trigger GRPO (跳过) ====="
else
    echo ""
    echo "===== Stage 3: Trigger GRPO ====="
    STAGE_START=$(date +%s)

    if [ -z "${LOAD_WEAVER_PATH}" ]; then
        echo "[pipeline] 错误: 未找到 Weaver checkpoint (需要先运行 Stage 1/2 或设置 LOAD_MODEL_PATH)"
        exit 1
    fi
    echo "[pipeline] 加载 Weaver: ${LOAD_WEAVER_PATH}"

    python -m accelerate.commands.launch \
        --config_file=configs/zero2.yaml \
        --num_processes=${NUM_GPUS} \
        main.py \
        --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
        --options \
        model.model_name ${REASONER_MODEL} \
        model.load_model_path ${LOAD_WEAVER_PATH} \
        model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
        model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
        model.weaver.model_name ${WEAVER_MODEL} \
        model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
        model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
        model.trigger.model_name ${TRIGGER_MODEL} \
        model.trigger.active True \
        datasets.mode grpo \
        run.mode train \
        run.train_weaver False \
        run.train_trigger True \
        run.train_trigger_method grpo \
        run.trigger.grpo.per_device_train_batch_size ${TRIGGER_BATCH_SIZE} \
        run.trigger.grpo.per_device_eval_batch_size ${TRIGGER_BATCH_SIZE} \
        run.trigger.grpo.num_train_epochs 1 \
        run.trigger.grpo.num_generations 8 \
        run.trigger.grpo.gradient_accumulation_steps ${TRIGGER_GRAD_ACCUM}

    LOAD_TRIGGER_PATH=$(find_latest_checkpoint)
    if [ -z "${LOAD_TRIGGER_PATH}" ]; then
        echo "[pipeline] 错误: Stage 3 未找到 checkpoint"
        exit 1
    fi

    STAGE_END=$(date +%s)
    echo "[pipeline] Stage 3 耗时: $((STAGE_END - STAGE_START))s"
    echo "[pipeline] Trigger GRPO checkpoint: ${LOAD_TRIGGER_PATH}"
fi

# ###########################################################################
#  Stage 4: 评测
# ###########################################################################
if [ "${SKIP_EVAL}" = "1" ]; then
    echo ""
    echo "===== Stage 4: 评测 (跳过) ====="
else
    echo ""
    echo "===== Stage 4: 评测 ====="
    STAGE_START=$(date +%s)

    # 优先使用 Trigger checkpoint，否则用 Weaver checkpoint
    EVAL_MODEL_PATH="${LOAD_TRIGGER_PATH:-${LOAD_WEAVER_PATH}}"
    if [ -z "${EVAL_MODEL_PATH}" ]; then
        echo "[pipeline] 错误: 未找到可评测的 checkpoint"
        exit 1
    fi
    echo "[pipeline] 评测模型: ${EVAL_MODEL_PATH}"

    python -m accelerate.commands.launch \
        --config_file=configs/zero2.yaml \
        --num_processes=${NUM_GPUS} \
        main.py \
        --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
        --options \
        model.model_name ${REASONER_MODEL} \
        model.load_model_path ${EVAL_MODEL_PATH} \
        model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
        model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
        model.weaver.model_name ${WEAVER_MODEL} \
        model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
        model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
        model.trigger.model_name ${TRIGGER_MODEL} \
        model.trigger.active ${TRIGGER_ACTIVE:-False} \
        run.mode evaluate \
        run.interaction.batch_size ${EVAL_BATCH_SIZE} \
        run.interaction.temperature 0.0 \
        run.interaction.max_response_length 1024

    STAGE_END=$(date +%s)
    echo "[pipeline] Stage 4 耗时: $((STAGE_END - STAGE_START))s"
fi

# ===== 完成 =====
PIPELINE_END=$(date +%s)
echo ""
echo "=============================="
echo "  MemGen Pipeline 完成"
echo "  总耗时: $((PIPELINE_END - PIPELINE_START))s"
echo "=============================="
