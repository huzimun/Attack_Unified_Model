#!/bin/bash

# Script to run batch image generation with Qwen-Image model
# Usage: ./run_batch_generation.sh [options]

# Set default values
DATA_NAME="final_order_prompt_policy"
export DATA_PATH="./datasets/${DATA_NAME}.json"
export OUTPUT_DIR="./outputs/generated_images/qwen-image_t2i_${DATA_NAME}"
export MODEL_NAME="Qwen/Qwen-Image"
export DEVICE="cuda:3"
export IMAGES_PER_PROMPT=1
export NUM_INFERENCE_STEPS=50
export TRUE_CFG_SCALE=4.0
# export SEED=42
# export ASPECT_RATIO="1:1"

python3 ./generation/Qwen-Image/src/safety_evaluation/new_batch_test_qwen-image_text2image.py \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --model_name $MODEL_NAME \
    --device $DEVICE \
    --images_per_prompt $IMAGES_PER_PROMPT \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --true_cfg_scale $TRUE_CFG_SCALE
    # --seed $SEED
    # --aspect_ratio $ASPECT_RATIO
