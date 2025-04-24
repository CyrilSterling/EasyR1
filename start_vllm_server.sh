#!/bin/bash

# Default model path - update this if your model is in a different location
MODEL_PATH=${1:-"/root/autodl-tmp/pretrained_models/Qwen2.5-1.5B-Instruct"}
PORT=${2:-8000}
API_KEY=${3:-"dummy-key-for-vllm"}
CUDA_DEVICES=${4:-"0"}

echo "Starting vLLM server with model: $MODEL_PATH on port $PORT with API key: $API_KEY"
echo "Using CUDA devices: $CUDA_DEVICES"

# Start the vLLM server with API key
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES vllm serve $MODEL_PATH \
    --dtype auto \
    --port $PORT \
    --api-key $API_KEY

# To use this script:
# ./start_vllm_server.sh [MODEL_PATH] [PORT] [API_KEY] [CUDA_DEVICES]
# Example: ./start_vllm_server.sh /path/to/model 8888 my-api-key 0,1 