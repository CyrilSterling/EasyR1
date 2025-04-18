#!/bin/bash
# upload_checkpoints.sh
# This script uploads checkpoints for multiple models based on pre-defined arrays.
# Each model may have multiple checkpoints at different global steps.
# Customize the associative array 'model_steps' with your model names as keys and their
# corresponding checkpoint steps as space-separated values.

# Define an associative array mapping each model to its checkpoints.
declare -A model_steps
# model_steps["qwen25vl-7b-grpo-math6k_wo_GPT_cos2"]="50 100 165"   
# model_steps["qwen25vl-7b-grpo-math6k_wo_GPT_cos1"]="50 100 165"                     
# model_steps["qwen25vl-7b-grpo-math6k_4096_cos1"]="50 100 165"     
# model_steps["qwen25vl-7b-grpo-math6k_4096_cos2"]="50 100 165"      
model_steps["qwen25vl-7b-grpo-math6k_4096_cos2_rollout32"]="50 100 165"                  

# Iterate over each model and its corresponding checkpoints.
for model in "${!model_steps[@]}"; do
  for step in ${model_steps[$model]}; do
  
    # Construct the local directory based on your directory structure.
    LOCAL_DIR="./outputs/${model}/global_step_${step}/actor"
    
    # Construct the Hugging Face upload path based on naming convention.
    HF_UPLOAD_PATH="mm-o1/${model}_step_${step}"
    
    # Print the upload details for tracking.
    echo "Uploading checkpoint from '${LOCAL_DIR}' to '${HF_UPLOAD_PATH}'..."
    
    # Run the upload command in background.
    python scripts/model_merger.py --local_dir "${LOCAL_DIR}" --hf_upload_path "${HF_UPLOAD_PATH}"
  done
done

# Wait for all parallel uploads to finish.
wait
echo "All uploads completed."