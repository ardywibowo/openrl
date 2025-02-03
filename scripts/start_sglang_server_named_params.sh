#!/bin/bash

# Default values for parameters
GPU_IDX=0
GPU_MEM_UTILIZATION=0.9

# Parse named parameters
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model-path) MODEL="$2"; shift ;;
        --port) PORT="$2"; shift ;;
        --random-seed) SEED="$2"; shift ;;
        --cpu-offload-gb) SWAP_SPACE="$2"; shift ;;
        --mem-fraction-static) GPU_MEM_UTILIZATION="$2"; shift ;;
        --gpu-idx) GPU_IDX="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

CUDA_VISIBLE_DEVICES=$GPU_IDX python -m sglang.launch_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --random-seed "$SEED" \
    --cpu-offload-gb "$SWAP_SPACE" \
    --mem-fraction-static "$GPU_MEM_UTILIZATION" \
    --dtype bfloat16
