set -x

# Load environment variables from .env file
if [ -f "$(dirname "$0")/../.env" ]; then
  echo "Loading environment variables from .env file"
  export $(grep -v '^#' "$(dirname "$0")/../.env" | xargs)
else
  echo "Warning: .env file not found"
fi

MODEL_PATH=/root/autodl-tmp/pretrained_models/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

# Set the checkpoint path to resume from
# Replace this with your actual checkpoint path if you want to resume from a specific checkpoint
# Example: CHECKPOINT_PATH=/root/autodl-tmp/checkpoints/easy_r1/qwen2_5_vl_3b_geo_grpo/global_step_1000

# Set custom checkpoint directory
SAVE_CHECKPOINT_PATH=/root/autodl-tmp/checkpoints

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/root/autodl-tmp/data/hiyouga/geometry3k@train \
    data.val_files=/root/autodl-tmp/data/hiyouga/geometry3k@test \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_3b_geo_grpo \
    trainer.n_gpus_per_node=2 \
    trainer.save_checkpoint_path=${SAVE_CHECKPOINT_PATH}
