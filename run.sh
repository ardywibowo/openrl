#!/bin/bash

source ./private/keys.sh

export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
CONFIG_NAME="experiments/vineppo/polIter_deepseekSft2_vineppo_MATH"
# CONFIG_NAME="experiments/vineppo/sft_deepseekmath_for_MATH"
# CONFIG_NAME="experiments/vineppo/polIter_deepseekSft2_ppo_MATH"
# CONFIG_NAME="experiments/linguistic_calibration/sft_llama2_for_paragraph_generation_claude_distill"
# CONFIG_NAME="experiments/linguistic_calibration/sft_llama2_for_answer_extraction_claude_distill"
# CONFIG_NAME="experiments/linguistic_calibration/sft_llama2_for_probability_forecasting_claude_distill"
# CONFIG_NAME="experiments/reward_modeling/reward_modeling_llama2"
# CONFIG_NAME="experiments/reward_modeling/reward_modeling_llama2_value_head_finetune"

CONFIGSTR="configs/${CONFIG_NAME}.jsonnet"
APP_DIRECTORY="experiments/${CONFIG_NAME}"

# Run ID = Config Name - Current Time
CURR_TIME=$(date +'%Y-%m-%d_%H-%M-%S')

# WANDB Run Name is CONFIG_NAME but with / replaced with -
WANDB_RUN_NAME=$(echo $CONFIG_NAME | tr / -)
export WANDB_RUN_ID="${WANDB_RUN_NAME}-${CURR_TIME}"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Run the training
deepspeed --no_local_rank --num_gpus=$NUM_GPUS  \
         src/treetune/main.py --configs "$CONFIGSTR" \
            run_iteration_loop
