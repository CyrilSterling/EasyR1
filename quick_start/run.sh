#!/bin/bash

# Usage: bash run.sh --data.curriculum_mixture_ratio 0.3 --worker.rollout.n 16 --data.curriculum_rollout_n 8 --data.curriculum_update_freq 7

# Default values
DEFAULT_MIXTURE_RATIO=0.5
DEFAULT_ROLLOUT_N=32
DEFAULT_CURR_UPDATE_FREQ=0
DEFAULT_CURR_ROLLOUT_N=8

# Initialize variables with defaults
MIXTURE_RATIO=$DEFAULT_MIXTURE_RATIO
ROLLOUT_N=$DEFAULT_ROLLOUT_N
CURR_UPDATE_FREQ=$DEFAULT_CURR_UPDATE_FREQ
CURR_ROLLOUT_N=$DEFAULT_CURR_ROLLOUT_N


# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data.curriculum_mixture_ratio)
            MIXTURE_RATIO="$2"
            shift 2
            ;;
        --worker.rollout.n)
            ROLLOUT_N="$2"
            shift 2
            ;;
        --data.curriculum_update_freq)
            CURR_UPDATE_FREQ="$2"
            shift 2
            ;;
        --data.curriculum_rollout_n)
            CURR_ROLLOUT_N="$2"
            shift 2
            ;;

        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Set job name and model path
JOB_NAME=mmr1_learn08_bleu02_mixv7_15k_ep10_0818_wollm_mixture$(echo $MIXTURE_RATIO | sed 's/\.//g')_currfreq${CURR_UPDATE_FREQ}_currrollout${CURR_ROLLOUT_N}_rollout${ROLLOUT_N}
MODEL_PATH=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/hf_hub/mm-o1/mmo1-math-qwen2.5_vl_3b-sft_mmr1_sft_0503_v10_mathinstruct_onlygemini_ep5

echo "Starting training with the following parameters:"
echo "  data.curriculum_mixture_ratio: $MIXTURE_RATIO"
echo "  data.curriculum_update_freq: $CURR_UPDATE_FREQ"
echo "  worker.rollout.n: $ROLLOUT_N"
echo "  data.curriculum_rollout_n: $CURR_ROLLOUT_N"
echo "  Job name: $JOB_NAME"
echo ""

export WANDB_API_KEY=c11743360602ee2f806e007952cfacf545764e25 
echo "WANDB project: $WANDB_PROJECT"

# Run the training command
JOB_NAME=$JOB_NAME \
    MODEL_PATH=$MODEL_PATH \
    bash /mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/projects/EasyR1/examples/mmr1/qwen2_5_vl_3b_mmr1_pub8k_wollm.sh \
    data.train_files=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/hf_hub/mm-o1/v7_mix_15k \
    data.max_response_length=4096 \
    data.sampling_strategy=curriculum \
    'data.curriculum_metrics=[learnability,self_bleu_123]' \
    'data.curriculum_metric_weights=[0.5,0.5]' \
    data.curriculum_mixture_ratio=$MIXTURE_RATIO \
    data.curriculum_update_freq=$CURR_UPDATE_FREQ \
    data.val_files=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/hf_hub/mm-o1/math_vista_val \
    trainer.total_episodes=10 \
    "worker.rollout.n=$ROLLOUT_N" \
    "data.curriculum_rollout_n=$CURR_ROLLOUT_N" \
    trainer.nnodes=1 \
    trainer.val_freq=4 \
    trainer.save_checkpoint_path=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/projects/EasyR1/checkpoints/$JOB_NAME \
    trainer.project_name=$WANDB_PROJECT