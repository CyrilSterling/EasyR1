# mmr1_shuffle_mixv7_15k_ep15
JOB_NAME=mmr1_shuffle_mixv7_15k_ep10_0713_wollm \
    MODEL_PATH=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/hf_hub/mm-o1/mmo1-math-qwen2.5_vl_3b-sft_mmr1_sft_0503_v10_mathinstruct_onlygemini_ep5 \
    bash EasyR1/examples/mmr1/qwen2_5_vl_3b_mmr1_pub8k_wollm.sh \
    data.train_files=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/hf_hub/mm-o1/v7_mix_15k \
    data.max_response_length=4096 \
    data.sampling_strategy=shuffle \
    data.val_files=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/hf_hub/mm-o1/math_vista_val \
    trainer.total_episodes=10 \
    worker.rollout.n=32 \
    trainer.nnodes=1 \
    trainer.val_freq=4 \
    trainer.save_checkpoint_path=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/projects/EasyR1/checkpoints/{$JOB_NAME}

# # mmr1_shuffle_mixv5_15k_ep10
# JOB_NAME=mmr1_shuffle_mixv5_15k_ep10_0713_wollm \
#     MODEL_PATH=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/hf_hub/mm-o1/mmo1-math-qwen2.5_vl_3b-sft_mmr1_sft_0503_v10_mathinstruct_onlygemini_ep5 \
#     bash EasyR1/examples/mmr1/qwen2_5_vl_3b_mmr1_pub8k_wollm.sh \
#     data.train_files=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/hf_hub/mm-o1/v5_mix_15k \
#     data.max_response_length=4096 \
#     data.sampling_strategy=shuffle \
#     data.val_files=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/hf_hub/mm-o1/math_vista_val \
#     trainer.total_episodes=10 \
#     worker.rollout.n=32 \
#     trainer.nnodes=1 \
#     trainer.val_freq=4 \
#     trainer.save_checkpoint_path=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/projects/EasyR1/checkpoints/{$JOB_NAME}

# # mmr1_learn08_bleu02_mixv7_15k_ep10
# JOB_NAME=mmr1_learn08_bleu02_mixv7_15k_ep10_0713_wollm \
#     MODEL_PATH=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/hf_hub/mm-o1/mmo1-math-qwen2.5_vl_3b-sft_mmr1_sft_0503_v10_mathinstruct_onlygemini_ep5 \
#     bash EasyR1/examples/mmr1/qwen2_5_vl_3b_mmr1_pub8k_wollm.sh \
#     data.train_files=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/hf_hub/mm-o1/v7_mix_15k \
#     data.max_response_length=4096 \
#     'data.curriculum_metrics=[learnability,self_bleu_123]' \
#     'data.curriculum_metric_weights=[0.8,0.2]' \
#     data.curriculum_update_freq=8 \
#     data.val_files=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/hf_hub/mm-o1/math_vista_val \
#     trainer.total_episodes=10 \
#     worker.rollout.n=32 \
#     trainer.nnodes=1 \
#     trainer.val_freq=4 \
#     trainer.save_checkpoint_path=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/projects/EasyR1/checkpoints/{$JOB_NAME}

# # mmr1_learn08_bleu02_mixv5_15k_ep10
# JOB_NAME=mmr1_learn08_bleu02_mixv5_15k_ep10_0713_wollm \
#     MODEL_PATH=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/hf_hub/mm-o1/mmo1-math-qwen2.5_vl_3b-sft_mmr1_sft_0503_v10_mathinstruct_onlygemini_ep5 \
#     bash EasyR1/examples/mmr1/qwen2_5_vl_3b_mmr1_pub8k_wollm.sh \
#     data.train_files=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/hf_hub/mm-o1/v5_mix_15k \
#     data.max_response_length=4096 \
#     'data.curriculum_metrics=[learnability,self_bleu_123]' \
#     'data.curriculum_metric_weights=[0.8,0.2]' \
#     data.curriculum_update_freq=8 \
#     data.val_files=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/hf_hub/mm-o1/math_vista_val \
#     trainer.total_episodes=10 \
#     worker.rollout.n=32 \
#     trainer.nnodes=1 \
#     trainer.val_freq=4 \
#     trainer.save_checkpoint_path=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/projects/EasyR1/checkpoints/{$JOB_NAME}


# mmr1_learn08_bleu02_mixv7_15k_ep10
JOB_NAME=mmr1_learn08_bleu02_mixv7_15k_ep10_0713_wollm_mixture08 \
    MODEL_PATH=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/hf_hub/mm-o1/mmo1-math-qwen2.5_vl_3b-sft_mmr1_sft_0503_v10_mathinstruct_onlygemini_ep5 \
    bash EasyR1/examples/mmr1/qwen2_5_vl_3b_mmr1_pub8k_wollm.sh \
    data.train_files=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/hf_hub/mm-o1/v7_mix_15k \
    data.max_response_length=4096 \
    'data.curriculum_metrics=[learnability,self_bleu_123]' \
    'data.curriculum_metric_weights=[0.8,0.2]' \
    'data.curriculum_mixture_ratio=0.8' \
    data.curriculum_update_freq=8 \
    data.val_files=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/hf_hub/mm-o1/math_vista_val \
    trainer.total_episodes=10 \
    worker.rollout.n=32 \
    trainer.nnodes=1 \
    trainer.val_freq=4 \
    trainer.save_checkpoint_path=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/projects/EasyR1/checkpoints/{$JOB_NAME}


JOB_NAME=mmr1_learn08_bleu02_mixv7_15k_ep10_0713_wollm_mixture02 \
    MODEL_PATH=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/hf_hub/mm-o1/mmo1-math-qwen2.5_vl_3b-sft_mmr1_sft_0503_v10_mathinstruct_onlygemini_ep5 \
    bash EasyR1/examples/mmr1/qwen2_5_vl_3b_mmr1_pub8k_wollm.sh \
    data.train_files=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/hf_hub/mm-o1/v7_mix_15k \
    data.max_response_length=4096 \
    'data.curriculum_metrics=[learnability,self_bleu_123]' \
    'data.curriculum_metric_weights=[0.8,0.2]' \
    'data.curriculum_mixture_ratio=0.2' \
    data.curriculum_update_freq=8 \
    data.val_files=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/hf_hub/mm-o1/math_vista_val \
    trainer.total_episodes=10 \
    worker.rollout.n=32 \
    trainer.nnodes=1 \
    trainer.val_freq=4 \
    trainer.save_checkpoint_path=/mnt/bn/tns-sicheng-llm-ruby-1/jiaxi.li/projects/EasyR1/checkpoints/{$JOB_NAME}