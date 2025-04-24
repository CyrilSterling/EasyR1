#!/bin/bash

# Default model path - update this if your model is in a different location
MODEL_PATH=${1:-"/mnt/ai4sci_develop_hz/sicong/checkpoints/Qwen2.5-VL-3B-Instruct"}
PORT=${2:-8000}
API_KEY=${3:-"dummy-key-for-vllm"}

echo "Starting vLLM server with model: $MODEL_PATH on port $PORT with API key: $API_KEY"

# Start the vLLM server with API key
CUDA_VISIBLE_DEVICES=7 vllm serve $MODEL_PATH \
    --dtype auto \
    --port $PORT \
    --api-key $API_KEY

# To use this script:
# ./start_vllm_server.sh [MODEL_PATH] [PORT] [API_KEY]
# Example: ./start_vllm_server.sh /path/to/model 8888 my-api-key 