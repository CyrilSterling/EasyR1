source ../.venv/bin/activate
set -x

which python

export VLLM_WORKER_MULTIPROC_METHOD=spawn

export HF_HUB_CACHE=/mnt/amlfs-02/shared/ckpts/
export HF_HUB_OFFLINE=1

tmux kill-session -t "ray"
tmux kill-session -t "judge"
tmux new-session -d -s "judge"

for i in {0..7}; do
    PORT=$((30000 + i))
    REMOTE_CMD="CUDA_VISIBLE_DEVICES=$i HF_HUB_CACHE=/mnt/amlfs-02/shared/ckpts/ HF_HUB_OFFLINE=1 \
        python -m vllm.entrypoints.openai.api_server \
        --served-model-name Qwen2.5-7B-Instruct \
        --model Qwen/Qwen2.5-7B-Instruct \
        -tp 1 \
        --port $PORT \
        --trust-remote-code \
        --max-model-len 32768 \
        --max-num-seqs 64 \
        --api-key EMPTY; sleep infinity"

    echo "Creating window judge-$i"
    tmux new-window -a -t "judge" -n "judge-$i" "$REMOTE_CMD"
done

mkdir -p /mnt/amlfs-01/home/jingwang/PROJECTS/mmo1/judge_ips/
hostname -I > /mnt/amlfs-01/home/jingwang/PROJECTS/mmo1/judge_ips/$WORKFLOW_ID