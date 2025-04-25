#!/bin/bash

# Fixed model path
MODEL_PATH="/mnt/ai4sci_develop_hz/sicong/checkpoints/Qwen2.5-VL-3B-Instruct" # Qwen2.5-7b-Instruct
# Port can be specified as the first argument, default is 8000
PORT=${1:-8000}
# Fixed API key
API_KEY="dummy-key-for-vllm"
# CUDA devices can be specified as the second argument, default is "0"
CUDA_DEVICES=${2:-"0"}

echo "Starting vLLM server with model: $MODEL_PATH on port $PORT with API key: $API_KEY"
echo "Using CUDA devices: $CUDA_DEVICES"

# Start the vLLM server with API key
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES vllm serve $MODEL_PATH \
    --dtype auto \
    --port $PORT \
    --api-key $API_KEY

# To use this script:
# ./start_vllm_server.sh [port] [cuda_device]
# Example: ./start_vllm_server.sh 8000 4