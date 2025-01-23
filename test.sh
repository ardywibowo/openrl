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

# Create a logs directory if missing
mkdir -p logs

for CONFIG_NAME in "${CONFIG_NAMES[@]}"; do
  CONFIGSTR="configs/${CONFIG_NAME}.jsonnet"
  
  # For clarity, remove any slashes in the config name to create a log filename
  WANDB_RUN_NAME=$(echo "$CONFIG_NAME" | tr '/' '-')
  CURRENT_TIME=$(date +'%Y-%m-%d_%H-%M-%S')
  LOG_FILE="logs/${WANDB_RUN_NAME}_${CURRENT_TIME}.log"

  # Build a custom WANDB run ID so we can see it in logs
  export WANDB_RUN_ID="${WANDB_RUN_NAME}-${CURRENT_TIME}"

  # Number of GPUs
  NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

  echo
  echo "=============================================================="
  echo "Starting job for $CONFIG_NAME"
  echo "Log file: $LOG_FILE"
  echo "=============================================================="

  # Record the start time in seconds
  START_TIME=$(date +%s)

  # ----------------------------------------------------------------
  # 1) Run the actual training in the background, capturing logs
  #    in real time. We also fork a "timeout killer" that will
  #    forcibly kill the job after TIMEOUT_DURATION. We will
  #    consider that a "pass".
  # ----------------------------------------------------------------

  # Run the training in the background, redirecting output to log
  deepspeed --no_local_rank --num_gpus="$NUM_GPUS" \
            src/treetune/main.py --configs "$CONFIGSTR" \
            run_iteration_loop \
            > "$LOG_FILE" 2>&1 &
  TRAIN_PID=$!

  # Tail logs so you can see them in real time
  tail -f "$LOG_FILE" &
  TAIL_PID=$!

  # Timeout killer. If job still runs after TIMEOUT_DURATION, kill it
  (
    sleep "$TIMEOUT_DURATION"
    if kill -0 "$TRAIN_PID" 2>/dev/null; then
      echo "[Timeout] Killing training job for $CONFIG_NAME after $TIMEOUT_DURATION seconds."
      kill -9 "$TRAIN_PID" 2>/dev/null
    fi
  ) &
  TIMEOUT_KILLER_PID=$!

  # ----------------------------------------------------------------
  # 2) Wait for the training job to exit, capturing its exit code.
  #    Then kill the log tail and the "timeout killer" process,
  #    as we no longer need them.
  # ----------------------------------------------------------------
  wait "$TRAIN_PID" 2>/dev/null
  TRAIN_EXIT_CODE=$?

  kill -9 "$TAIL_PID" 2>/dev/null
  kill -9 "$TIMEOUT_KILLER_PID" 2>/dev/null

  END_TIME=$(date +%s)
  ELAPSED=$(( END_TIME - START_TIME ))

# /mnt/task_runtime/experiments/polIter_deepseekSft2_ppo_MATH_20250123_095620/temp_episodes/iteration__0000/episodes/merged/dataset_info.json
  # ----------------------------------------------------------------
  # 3) Decide pass/fail:
  #    - If the job ended with code=0 before the timeout => PASS
  #    - If the job ended with code!=0 before the timeout => FAIL
  #    - If it got forcibly killed at TIMEOUT => PASS
  # ----------------------------------------------------------------
  if [ "$ELAPSED" -ge "$TIMEOUT_DURATION" ]; then
    # Ran the full duration or was forcibly killed at the end => pass
    echo "Training job for $CONFIG_NAME ran to the ${TIMEOUT_DURATION}s limit. PASS."
  else
    # It finished before the timeout
    if [ "$TRAIN_EXIT_CODE" -eq 0 ]; then
      echo "Training job for $CONFIG_NAME completed early with exit code 0. PASS."
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
