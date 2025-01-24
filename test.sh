#!/bin/bash

# Activate conda environment
source ./private/keys.sh

# Timeout duration in seconds
TIMEOUT_DURATION=450  # Set the timeout in seconds

# List of configuration names
CONFIG_NAMES=(
  "experiments/vineppo/polIter_deepseekSft2_vineppo_MATH"
  "experiments/vineppo/sft_deepseekmath_for_MATH"
  "experiments/vineppo/polIter_deepseekSft2_ppo_MATH"
  "experiments/linguistic_calibration/sft_llama2_for_paragraph_generation_claude_distill"
  "experiments/linguistic_calibration/sft_llama2_for_answer_extraction_claude_distill"
  "experiments/linguistic_calibration/sft_llama2_for_probability_forecasting_claude_distill"
  "experiments/reward_modeling/reward_modeling_llama2"
  "experiments/reward_modeling/reward_modeling_llama2_value_head_finetune"
)


# Keep track of any failures
FAILED_TESTS=()

mkdir -p logs

for CONFIG_NAME in "${CONFIG_NAMES[@]}"; do
  CONFIGSTR="configs/${CONFIG_NAME}.jsonnet"

  WANDB_RUN_NAME=$(echo "$CONFIG_NAME" | tr '/' '-')
  CURRENT_TIME=$(date +'%Y-%m-%d_%H-%M-%S')
  LOG_FILE="logs/${WANDB_RUN_NAME}_${CURRENT_TIME}.log"

  export WANDB_RUN_ID="${WANDB_RUN_NAME}-${CURRENT_TIME}"

  NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

  echo
  echo "=============================================================="
  echo "Starting job for $CONFIG_NAME"
  echo "Log file: $LOG_FILE"
  echo "=============================================================="

  START_TIME=$(date +%s)

  # Ensure the log file exists (so tail won't complain)
  touch "$LOG_FILE"

  # 1) Start training as a background job, with output redirected to our log file.
  deepspeed --no_local_rank --num_gpus="$NUM_GPUS" \
            src/treetune/main.py --configs "$CONFIGSTR" \
            run_iteration_loop \
            > "$LOG_FILE" 2>&1 &
  TRAIN_PID=$!

  # 2) Start tailing the log in the background, so we can see the logs in real time.
  tail -f "$LOG_FILE" &
  TAIL_PID=$!

  # 3) Timeout process to kill training if it runs longer than TIMEOUT_DURATION.
  (
    sleep "$TIMEOUT_DURATION"
    if kill -0 "$TRAIN_PID" 2>/dev/null; then
      echo "[Timeout] Killing training job for $CONFIG_NAME after $TIMEOUT_DURATION seconds."
      kill -9 "$TRAIN_PID" 2>/dev/null
    fi
  ) &
  TIMEOUT_KILLER_PID=$!

  # 4) Wait for the training job to exit, capturing its exit code.
  wait "$TRAIN_PID" 2>/dev/null
  TRAIN_EXIT_CODE=$?

  # Kill the tail and the timeout‐killer processes, as they're no longer needed.
  kill -9 "$TAIL_PID" 2>/dev/null
  kill -9 "$TIMEOUT_KILLER_PID" 2>/dev/null

  END_TIME=$(date +%s)
  ELAPSED=$(( END_TIME - START_TIME ))

  if [ "$ELAPSED" -ge "$TIMEOUT_DURATION" ]; then
    # The job ran or was forced to run until the timeout => PASS
    echo "Training job for $CONFIG_NAME ran to ${TIMEOUT_DURATION}s. PASS."
  else
    # If it ended before the timeout, check the exit code
    if [ "$TRAIN_EXIT_CODE" -eq 0 ]; then
      echo "Training job for $CONFIG_NAME completed (exit code 0). PASS."
    else
      echo "Training job for $CONFIG_NAME failed early (code: $TRAIN_EXIT_CODE). FAIL."
      FAILED_TESTS+=("$CONFIG_NAME")
    fi
  fi

  echo "Killing any leftover Python processes..."
  ps aux | grep python | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null
  echo "----------------------------------------------------"
done

# Final report
if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
  echo "All tests passed (including timeouts)!"
else
  echo "The following tests failed before the timeout:"
  for TEST in "${FAILED_TESTS[@]}"; do
    echo "  - $TEST"
  done
fi
