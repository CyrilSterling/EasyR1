# mmr1_shuffle_mixv7_15k_ep15
JOB_NAME=mmr1_shuffle_mixv7_15k_ep10_0713 \
    MODEL_PATH=/mnt/amlfs-03/shared/jingwang/PROJECTS/mmo1/finetune/LLaMA-Factory/saves/mmo1-math-qwen2.5_vl_3b-sft_mmr1_sft_0503_v10_mathinstruct_onlygemini_ep5 \
    bash EasyR1/examples/mmr1/qwen2_5_vl_7b_mmr1_pub8k_wllm.sh \
    data.train_files=/mnt/amlfs-02/shared/ckpts/mmn/datasets--mm-o1--v7_mix_15k/snapshots/1308585089135e77cf542923497d6c6351dff979/ \
    data.max_response_length=4096 \
    data.sampling_strategy=shuffle \
    data.val_files=/mnt/amlfs-02/shared/ckpts/mmn/datasets--mm-o1--math_vista_val/snapshots/6d4db5f646fc7a5a4afe5599208870a929b38722/ \
    trainer.total_episodes=10 \
    worker.rollout.n=32 \
    trainer.nnodes=8 \
    trainer.val_freq=4 \
    trainer.save_checkpoint_path=/mnt/amlfs-02/shared/checkpoints/mmn/saves/{$JOB_NAME}

# mmr1_shuffle_mixv5_15k_ep10
JOB_NAME=mmr1_shuffle_mixv5_15k_ep10_0713 \
    MODEL_PATH=/mnt/amlfs-03/shared/jingwang/PROJECTS/mmo1/finetune/LLaMA-Factory/saves/mmo1-math-qwen2.5_vl_3b-sft_mmr1_sft_0503_v10_mathinstruct_onlygemini_ep5 \
    bash EasyR1/examples/mmr1/qwen2_5_vl_7b_mmr1_pub8k_wllm.sh \
    data.train_files=/mnt/amlfs-02/shared/ckpts/mmn/datasets--mm-o1--v5_mix_15k/snapshots/d476d8a8624c647c08cddbda99c00bbd027f7f7b/ \
    data.max_response_length=4096 \
    data.sampling_strategy=shuffle \
    data.val_files=/mnt/amlfs-02/shared/ckpts/mmn/datasets--mm-o1--math_vista_val/snapshots/6d4db5f646fc7a5a4afe5599208870a929b38722/ \
    trainer.total_episodes=10 \
    worker.rollout.n=32 \
    trainer.nnodes=8 \
    trainer.val_freq=4 \
    trainer.save_checkpoint_path=/mnt/amlfs-02/shared/checkpoints/mmn/saves/{$JOB_NAME}

# mmr1_learn08_bleu02_mixv7_15k_ep10
JOB_NAME=mmr1_learn08_bleu02_mixv7_15k_ep10_0713 \
    MODEL_PATH=/mnt/amlfs-03/shared/jingwang/PROJECTS/mmo1/finetune/LLaMA-Factory/saves/mmo1-math-qwen2.5_vl_3b-sft_mmr1_sft_0503_v10_mathinstruct_onlygemini_ep5 \
    bash EasyR1/examples/mmr1/qwen2_5_vl_7b_mmr1_pub8k_wllm.sh \
    data.train_files=/mnt/amlfs-02/shared/ckpts/mmn/datasets--mm-o1--v7_mix_15k/snapshots/1308585089135e77cf542923497d6c6351dff979/ \
    data.max_response_length=4096 \
    'data.curriculum_metrics=[learnability,self_bleu_123]' \
    'data.curriculum_metric_weights=[0.8,0.2]' \
    data.curriculum_update_freq=8 \
    data.val_files=/mnt/amlfs-02/shared/ckpts/mmn/datasets--mm-o1--math_vista_val/snapshots/6d4db5f646fc7a5a4afe5599208870a929b38722/ \
    trainer.total_episodes=10 \
    worker.rollout.n=32 \
    trainer.nnodes=8 \
    trainer.val_freq=4 \
    trainer.save_checkpoint_path=/mnt/amlfs-02/shared/checkpoints/mmn/saves/{$JOB_NAME}

# mmr1_learn08_bleu02_mixv5_15k_ep10
JOB_NAME=mmr1_learn08_bleu02_mixv5_15k_ep10_0713 \
    MODEL_PATH=/mnt/amlfs-03/shared/jingwang/PROJECTS/mmo1/finetune/LLaMA-Factory/saves/mmo1-math-qwen2.5_vl_3b-sft_mmr1_sft_0503_v10_mathinstruct_onlygemini_ep5 \
    bash EasyR1/examples/mmr1/qwen2_5_vl_7b_mmr1_pub8k_wllm.sh \
    data.train_files=/mnt/amlfs-02/shared/ckpts/mmn/datasets--mm-o1--v5_mix_15k/snapshots/d476d8a8624c647c08cddbda99c00bbd027f7f7b/ \
    data.max_response_length=4096 \
    'data.curriculum_metrics=[learnability,self_bleu_123]' \
    'data.curriculum_metric_weights=[0.8,0.2]' \
    data.curriculum_update_freq=8 \
    data.val_files=/mnt/amlfs-02/shared/ckpts/mmn/datasets--mm-o1--math_vista_val/snapshots/6d4db5f646fc7a5a4afe5599208870a929b38722/ \
    trainer.total_episodes=10 \
    worker.rollout.n=32 \
    trainer.nnodes=8 \
    trainer.val_freq=4 \
    trainer.save_checkpoint_path=/mnt/amlfs-02/shared/checkpoints/mmn/saves/{$JOB_NAME}