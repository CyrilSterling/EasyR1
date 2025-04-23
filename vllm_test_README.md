# Testing vLLM as a Judge for Reward Calculation

This guide explains how to test the vLLM model serving as a judge for the reward calculation system.

## Prerequisites

- EasyR1 repository
- vLLM installed
- Access to a compatible model (e.g., Qwen2.5-1.5B-Instruct)

## Step 1: Start the vLLM Server

First, start the vLLM server with your chosen model:

```bash
./start_vllm_server.sh [MODEL_PATH] [PORT]
```

Example:
```bash
./start_vllm_server.sh /root/autodl-tmp/pretrained_models/Qwen2.5-1.5B-Instruct 8000 "dummy-key-for-vllm"
```

The server will start and display log messages. Wait until you see "vLLM API server started" before proceeding.

## Step 2: Run the Standalone Test Script

The standalone test script provides a simple way to test the vLLM judge functionality:

```bash
python test_vllm_judge.py --base_url "http://localhost:8000/v1" --model_name /root/autodl-tmp/pretrained_models/Qwen2.5-1.5B-Instruct --api_key "dummy-key-for-vllm"
```

This script will:
1. Call the vLLM server with some test examples
2. Print out the reward scores for each example
3. Test direct interaction with the judge

When you run the script, it will pause at the breakpoint we inserted. You can inspect the variables by:
- Continuing execution: type `c` and press Enter
- Step through code: type `n` for next line
- Print variable values: type variable names
- Quit debugger: type `q`

## Step 3: Test with Full Training Pipeline

To test with the full training pipeline:

```bash
python verl/trainer/main.py --config test_vllm_config.yaml
```

This will run a test validation using the vLLM judge for reward calculation.

## Debugging Tips

- Breakpoints have been added to key functions for debugging
- Look for the log message "Corrected X/Y rewards using vLLM as judge" to confirm it's working
- Check server logs for API calls
- If issues occur, try adjusting the model parameters or prompt format

## Configuration Options

In `test_vllm_config.yaml`, you can modify:
- `base_url`: URL of your vLLM server
- `model_name`: Name of the model to use
- `cos_len_reward_config`: Parameters for reward calculation 