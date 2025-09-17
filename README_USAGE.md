# EasyR1 Training Guide

This guide provides comprehensive instructions for launching RLHF training with EasyR1, based on the examples in `scripts/submit.fish`.

## Prerequisites

1. **Virtual Environment Setup**
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

2. **Environment Variables**
Create a `.env` file in the project root (optional):
```bash
# .env
MODEL_PATH=/path/to/default/model
HF_ENDPOINT=https://hf-mirror.com  # Optional: for Hugging Face mirror
```

## Training Strategies

EasyR1 supports three main sampling strategies for training:

### 1. Shuffle Strategy (Random Sampling)

The simplest approach - randomly samples from the dataset:

```bash
JOB_NAME=my_shuffle_training \
MODEL_PATH=/path/to/your/model \
bash EasyR1/examples/mmr1/qwen2_5_vl_3b_mmr1_pub8k_wollm.sh \
    data.train_files=/path/to/training/data \
    data.max_response_length=4096 \
    data.sampling_strategy=shuffle \
    data.val_files=/path/to/validation/data \
    trainer.total_episodes=10 \
    worker.rollout.n=32 \
    trainer.nnodes=1 \
    trainer.val_freq=4 \
    trainer.save_checkpoint_path=/path/to/checkpoints/${JOB_NAME}
```

### 2. Sequential Strategy

Processes samples in order, useful for debugging or specific ordering requirements:

```bash
JOB_NAME=my_sequential_training \
MODEL_PATH=/path/to/your/model \
bash EasyR1/examples/mmr1/qwen2_5_vl_3b_mmr1_pub8k_wollm.sh \
    data.train_files=/path/to/training/data \
    data.sampling_strategy=sequential \
    trainer.total_episodes=10 \
    trainer.save_checkpoint_path=/path/to/checkpoints/${JOB_NAME}
```

### 3. Curriculum Learning Strategy

Advanced strategy that dynamically weights samples based on difficulty metrics:

#### Basic Curriculum with Learnability
```bash
JOB_NAME=my_curriculum_basic \
MODEL_PATH=/path/to/your/model \
bash EasyR1/examples/mmr1/qwen2_5_vl_3b_mmr1_pub8k_wollm.sh \
    data.train_files=/path/to/training/data \
    data.sampling_strategy=curriculum \
    'data.curriculum_metrics=[learnability]' \
    'data.curriculum_metric_weights=[1.0]' \
    data.curriculum_update_freq=8 \
    data.curriculum_mixture_ratio=0.5 \
    trainer.total_episodes=10 \
    trainer.save_checkpoint_path=/path/to/checkpoints/${JOB_NAME}
```

#### Advanced Curriculum with Multiple Metrics
```bash
JOB_NAME=my_curriculum_advanced \
MODEL_PATH=/path/to/your/model \
bash EasyR1/examples/mmr1/qwen2_5_vl_3b_mmr1_pub8k_wollm.sh \
    data.train_files=/path/to/training/data \
    data.sampling_strategy=curriculum \
    'data.curriculum_metrics=[learnability,self-bleu]' \
    'data.curriculum_metric_weights=[0.8,0.2]' \
    data.curriculum_update_freq=8 \
    data.curriculum_mixture_ratio=0.8 \
    data.curriculum_rollout_n=8 \
    trainer.total_episodes=10 \
    trainer.save_checkpoint_path=/path/to/checkpoints/${JOB_NAME}
```

## Key Configuration Parameters

### Data Configuration
- `data.train_files`: Training dataset path
- `data.val_files`: Validation dataset path
- `data.max_response_length`: Maximum response length in tokens (default: 8192)
- `data.max_prompt_length`: Maximum prompt length in tokens (default: 2048)
- `data.rollout_batch_size`: Batch size for rollout generation (default: 2048)
- `data.sampling_strategy`: Choose from "shuffle", "sequential", or "curriculum"

### Curriculum-Specific Parameters
- `data.curriculum_metrics`: List of metrics for sample weighting
  - `learnability`: Measures sample difficulty based on model performance
  - `distinct`: N-gram diversity in generated responses
  - `self-bleu`: Similarity between generated responses
  - `edit-distance`: Pairwise edit distance between responses
- `data.curriculum_metric_weights`: Weights for combining multiple metrics
- `data.curriculum_mixture_ratio`: Ratio of weighted vs random sampling (0.0-1.0)
  - 0.0 = fully random sampling
  - 1.0 = fully weighted sampling
  - 0.5 = 50% weighted, 50% random
- `data.curriculum_update_freq`: Update weights every N steps (0 for epoch-level)
- `data.curriculum_rollout_n`: Number of rollouts per sample for metric calculation

### Training Configuration
- `trainer.total_episodes`: Number of training episodes
- `trainer.nnodes`: Number of nodes for distributed training
- `trainer.n_gpus_per_node`: GPUs per node (default: 8)
- `trainer.val_freq`: Validation frequency (-1 to disable)
- `trainer.save_freq`: Checkpoint save frequency
- `trainer.save_checkpoint_path`: Directory to save checkpoints
- `trainer.experiment_name`: Name for wandb/tensorboard tracking

### Worker Configuration
- `worker.rollout.n`: Number of response samples per prompt during training
- `worker.rollout.temperature`: Sampling temperature
- `worker.actor.global_batch_size`: Global batch size for training
- `worker.actor.optim.lr`: Learning rate

## Complete Example Configurations

### Example 1: Production Training with Curriculum Learning
```bash
JOB_NAME=production_curriculum_v1 \
MODEL_PATH=/models/qwen2.5-vl-3b-instruct \
bash EasyR1/examples/mmr1/qwen2_5_vl_3b_mmr1_pub8k_wollm.sh \
    data.train_files=/data/training/math_reasoning_15k \
    data.val_files=/data/validation/math_vista \
    data.max_response_length=4096 \
    data.sampling_strategy=curriculum \
    'data.curriculum_metrics=[learnability,self-bleu]' \
    'data.curriculum_metric_weights=[0.8,0.2]' \
    data.curriculum_mixture_ratio=0.8 \
    data.curriculum_update_freq=8 \
    data.curriculum_rollout_n=8 \
    trainer.total_episodes=10 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.val_freq=4 \
    trainer.save_freq=5 \
    trainer.save_checkpoint_path=/checkpoints/${JOB_NAME} \
    worker.rollout.n=32 \
    worker.rollout.temperature=0.6 \
    worker.actor.global_batch_size=512 \
    worker.actor.optim.lr=1.0e-6
```

### Example 2: Quick Testing with Shuffle
```bash
JOB_NAME=test_shuffle \
MODEL_PATH=/models/test_model \
bash EasyR1/examples/mmr1/qwen2_5_vl_3b_mmr1_pub8k_wollm.sh \
    data.train_files=/data/small_dataset \
    data.sampling_strategy=shuffle \
    trainer.total_episodes=2 \
    trainer.val_freq=-1 \
    worker.rollout.n=4 \
    trainer.save_checkpoint_path=/tmp/test_checkpoints
```

### Example 3: Multi-Node Distributed Training
```bash
JOB_NAME=distributed_training \
MODEL_PATH=/models/large_model \
bash EasyR1/examples/mmr1/qwen2_5_vl_3b_mmr1_pub8k_wollm.sh \
    data.train_files=/data/large_dataset \
    data.sampling_strategy=curriculum \
    'data.curriculum_metrics=[learnability]' \
    trainer.total_episodes=20 \
    trainer.nnodes=4 \
    trainer.n_gpus_per_node=8 \
    worker.actor.fsdp.fsdp_size=32 \
    trainer.save_checkpoint_path=/distributed_checkpoints/${JOB_NAME}
```

## Direct Python Usage

For more control, you can directly invoke the Python training script:

```python
python -m verl.trainer.main \
    config=EasyR1/examples/mmr1_b200.yaml \
    data.train_files=/path/to/data \
    data.system_prompt="Your custom system prompt" \
    worker.actor.model.model_path=/path/to/model \
    trainer.experiment_name=my_experiment \
    trainer.total_episodes=10 \
    data.sampling_strategy=curriculum \
    'data.curriculum_metrics=[learnability,distinct]' \
    'data.curriculum_metric_weights=[0.7,0.3]'
```

## Custom Configuration File

Create your own YAML configuration:

```yaml
# my_custom_config.yaml
defaults:
  - mmr1_b200  # Inherit from base config

data:
  train_files: /my/training/data
  val_files: /my/validation/data
  max_response_length: 4096
  sampling_strategy: curriculum
  curriculum_metrics:
    - learnability
    - self-bleu
  curriculum_metric_weights:
    - 0.7
    - 0.3
  curriculum_mixture_ratio: 0.6
  curriculum_update_freq: 10

trainer:
  total_episodes: 15
  n_gpus_per_node: 4
  experiment_name: my_custom_experiment

worker:
  rollout:
    n: 16
    temperature: 0.7
  actor:
    optim:
      lr: 5.0e-7
```

Then use it:
```bash
python -m verl.trainer.main config=my_custom_config.yaml
```

## Monitoring & Logging

Training progress is automatically logged to Weights & Biases:

```bash
# View metrics including:
# - Training loss and rewards
# - Curriculum weight distribution
# - Sample difficulty metrics
# - Validation performance
# - Generated sample quality

# Configure project and experiment names
trainer.project_name=my_project \
trainer.experiment_name=experiment_v1
```

## Resume Training

Resume from a checkpoint:

```bash
JOB_NAME=resume_training \
MODEL_PATH=/path/to/model \
bash EasyR1/examples/mmr1/qwen2_5_vl_3b_mmr1_pub8k_wollm.sh \
    trainer.load_checkpoint_path=/checkpoints/previous_run/global_step_100 \
    trainer.save_checkpoint_path=/checkpoints/${JOB_NAME} \
    trainer.total_episodes=20  # Continue for more episodes
```

## Tips for Different Scenarios

### For Small Datasets (<10K samples)
- Use smaller `curriculum_rollout_batch_size` (e.g., 256)
- Consider `curriculum_update_freq=0` for epoch-level updates
- Lower `curriculum_rollout_n` to reduce computation (e.g., 4)

### For Large Models (>30B parameters)
- Enable gradient checkpointing: `worker.actor.model.enable_gradient_checkpointing=true`
- Use CPU offloading if needed: `worker.actor.fsdp.enable_cpu_offload=true`
- Reduce batch sizes and increase gradient accumulation

### For Quick Experimentation
- Disable validation: `trainer.val_freq=-1`
- Use fewer rollouts: `worker.rollout.n=4`
- Save less frequently: `trainer.save_freq=10`

### For Production Deployment
- Enable all validation: `trainer.val_before_train=true`
- Use appropriate checkpoint limits: `trainer.save_limit=10`
- Monitor curriculum metrics closely
- Use higher `curriculum_rollout_n` for better metric estimation (e.g., 16-32)