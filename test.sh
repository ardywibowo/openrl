#!/bin/bash

# Activate conda environment
source ./private/keys.sh

export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1

# Timeout duration in minutes
TIMEOUT_DURATION=15

# Timeout duration in seconds
TIMEOUT_DURATION=$((TIMEOUT_DURATION * 60))

# List of configuration names
CONFIG_NAMES=(
  # "experiments/vineppo/polIter_deepseekSft2_vineppo_MATH"
  # "experiments/vineppo/sft_deepseekmath_for_MATH"
  "experiments/vineppo/polIter_deepseekSft2_ppo_MATH"
  "experiments/linguistic_calibration/sft_llama2_for_paragraph_generation_claude_distill"
  # "experiments/linguistic_calibration/sft_llama2_for_answer_extraction_claude_distill"
  # "experiments/linguistic_calibration/sft_llama2_for_probability_forecasting_claude_distill"
  "experiments/reward_modeling/reward_modeling_llama2"
  "experiments/reward_modeling/reward_modeling_llama2_no_value_head_finetune"
)

# Loop through each configuration
for CONFIG_NAME in "${CONFIG_NAMES[@]}"; do
  CONFIGSTR="configs/${CONFIG_NAME}.jsonnet"
  APP_DIRECTORY="experiments/${CONFIG_NAME}"

  # Run ID = Config Name - Current Time
  CURR_TIME=$(date +'%Y-%m-%d_%H-%M-%S')

  # WANDB Run Name is CONFIG_NAME but with / replaced with -
  WANDB_RUN_NAME=$(echo $CONFIG_NAME | tr / -)
  export WANDB_RUN_ID="${WANDB_RUN_NAME}-${CURR_TIME}"

  # Get the number of GPUs
  NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

  # Create a log file for the current config
  LOG_FILE="logs/${WANDB_RUN_NAME}.log"
  mkdir -p logs

  # Start the training process and redirect stdout/stderr to the log file
  deepspeed --no_local_rank --num_gpus=$NUM_GPUS \
            src/treetune/main.py --configs "$CONFIGSTR" \
            run_iteration_loop > "$LOG_FILE" 2>&1 &
  TRAINING_PID=$!

  echo "Started training job for $CONFIG_NAME with PID $TRAINING_PID. Logs: $LOG_FILE"

  # Monitor logs in real time while enforcing the timeout
  { sleep $TIMEOUT_DURATION; kill -9 $TRAINING_PID 2>/dev/null; } &
  TIMEOUT_PID=$!

  # Show logs in real time
  tail -f "$LOG_FILE" --pid=$TIMEOUT_PID

  # Check if the process was terminated due to timeout
  if ps -p $TRAINING_PID > /dev/null; then
    echo "Killing training job for $CONFIG_NAME (PID $TRAINING_PID) after timeout"
    kill -9 $TRAINING_PID
    echo "Training job for $CONFIG_NAME was terminated before reaching the timeout."
  else
    echo "Training job for $CONFIG_NAME ran successfully for the full $((TIMEOUT_DURATION / 60)) minutes."
  fi

  # Kill any lingering Python processes
  echo "Killing any lingering Python processes..."
  ps aux | grep python | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null

  echo "------------------------------------------"
done

echo "All training jobs completed."
