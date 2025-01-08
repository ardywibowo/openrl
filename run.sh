export WANDB_API_KEY="<WANDB_API_KEY>"
huggingface-cli login --token "<HF_TOKEN>"

export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
CONFIG_NAME="experiments/linguistic_calibration/sft_llama2_for_probability_forecasting_claude_distill"
CONFIGSTR="configs/${CONFIG_NAME}.jsonnet"
APP_DIRECTORY="experiments/${CONFIG_NAME}"

export APP_SEED="2746318213"

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
