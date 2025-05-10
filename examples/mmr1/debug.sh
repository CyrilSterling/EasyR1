# Activate the virtual environment
if [ -f "../.venv/bin/activate" ]; then
  source ../.venv/bin/activate
else
  echo "Warning: ../.venv/bin/activate not found"
fi

set -x
which python

# Load environment variables from .env file
if [ -f "$(dirname "$0")/../.env" ]; then
  echo "Loading environment variables from .env file"
  export $(grep -v '^#' "$(dirname "$0")/../.env" | xargs)
else
  echo "Warning: .env file not found"
fi

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct # replace it with your local file path
echo $MODEL_PATH

SYSTEM_PROMPT="""A conversation between User and Assistant. 
The User provides an image and asks a question. 
The Assistant first analyzes both the image and the question, 
then carefully thinks about the reasoning process step by step, 
and finally provides the User with an accurate answer. 
The Assistant must carefully checkout the correctness and validity of each reasoning step. 
If any errors or inconsistencies are found during the reasoning process, 
the Assistant reflects and corrects them logically. 
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, 
i.e., <think> reasoning process here, with potential reflections and corrections </think>
<answer> final answer here, with the key result enclosed in \boxed{} </answer>."""

# assert JOB_NAME is not empty
if [ -z "$JOB_NAME" ]; then
  echo "JOB_NAME is empty"
  exit 1
fi

SUBMISSION_ID=${JOB_NAME}_$(date +%s)

RUNTIME_ENV=$(cat <<EOF | jq -c '.'
{
  "env_vars": {
    "HF_HUB_CACHE": "/mnt/amlfs-02/shared/ckpts/mmc/",
    "WANDB_API_KEY": "26dcd5fab9afa1c6f127a04db6e0af6521affbbb",
    "WANDB_ENTITY": "mmo1"
  }
}
EOF
)
mkdir -p /workspace/empty/

WORKDIR=$PWD

RAY_ADDRESS='http://127.0.0.1:8265' ray job submit \
  --working-dir /workspace/empty/ \
  --log-style pretty \
  --runtime-env-json "$RUNTIME_ENV" \
  --submission-id ${SUBMISSION_ID} \
  --no-wait \
  -- \
  cd $WORKDIR \; \
  python -m verl.trainer.main \
  config=$WORKDIR/examples/mmr1_batchvllm.yaml \
  data.system_prompt="${SYSTEM_PROMPT}" \
  data.train_files=/mnt/amlfs-01/home/jingwang/DATASET/mmo1/HF/rl_pub_8k/ \
  data.val_files=mm-o1/math_vista_val \
  worker.actor.model.model_path=${MODEL_PATH} \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=8 \
  trainer.experiment_name=${JOB_NAME} \
  trainer.save_checkpoint_path=/mnt/amlfs-02/shared/jingwang/checkpoints/mmc/${JOB_NAME} \
  worker.reward.workflow_id=${WORKFLOW_ID}
