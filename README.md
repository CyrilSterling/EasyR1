# EasyR1 Curriculum Learning Guide

This guide provides comprehensive instructions for using curriculum learning strategies in EasyR1, including advanced sampling techniques for RLHF training.

## Overview

EasyR1 supports three main sampling strategies for training:

- **Shuffle**: Random sampling from the dataset
- **Sequential**: Process samples in order
- **Curriculum**: Dynamic sample weighting based on difficulty metrics

## Quick Start

### Basic Training with Shuffle Strategy

```bash
JOB_NAME=my_training \
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct \
bash examples/mmr1/train_qwen2_5_vl_3b.sh \
    data.train_files=/path/to/training/data \
    data.sampling_strategy=shuffle \
    trainer.total_episodes=10
```

### Curriculum Learning with Learnability Metric

```bash
JOB_NAME=curriculum_training \
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct \
bash examples/mmr1/train_qwen2_5_vl_3b.sh \
    data.train_files=/path/to/training/data \
    data.sampling_strategy=curriculum \
    'data.curriculum_metrics=[learnability]' \
    'data.curriculum_metric_weights=[1.0]' \
    data.curriculum_mixture_ratio=0.5 \
    trainer.total_episodes=10
```

## Curriculum Learning Features

### Available Metrics

- **learnability**: Measures sample difficulty based on model performance
- **distinct**: N-gram diversity in generated responses
- **self-bleu**: Similarity between generated responses
- **edit-distance**: Pairwise edit distance between responses

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `curriculum_metrics` | List of metrics for weighting | `[learnability]` |
| `curriculum_metric_weights` | Weights for combining metrics | `[1.0]` |
| `curriculum_mixture_ratio` | Ratio of weighted vs random sampling (0.0-1.0) | `0.5` |
| `curriculum_update_freq` | Update weights every N steps (0 for epoch-level) | `4` |
| `curriculum_rollout_n` | Number of rollouts for metric calculation | `8` |
| `curriculum_momentum` | Momentum for weight updates | `0.0` |

## Advanced Examples

### Multi-Metric Curriculum Learning

Combine learnability and diversity metrics:

```bash
JOB_NAME=advanced_curriculum \
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct \
bash examples/mmr1/train_qwen2_5_vl_3b.sh \
    data.train_files=/path/to/data \
    data.sampling_strategy=curriculum \
    'data.curriculum_metrics=[learnability,self-bleu]' \
    'data.curriculum_metric_weights=[0.8,0.2]' \
    data.curriculum_mixture_ratio=0.8 \
    data.curriculum_update_freq=8 \
    trainer.total_episodes=10
```

### Production Training Configuration

```bash
JOB_NAME=production_v1 \
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
bash examples/mmr1/train_qwen2_5_vl_3b.sh \
    data.train_files=/data/large_dataset \
    data.max_response_length=4096 \
    data.sampling_strategy=curriculum \
    'data.curriculum_metrics=[learnability,distinct,self-bleu]' \
    'data.curriculum_metric_weights=[0.6,0.2,0.2]' \
    data.curriculum_mixture_ratio=0.7 \
    data.curriculum_update_freq=10 \
    data.curriculum_rollout_n=16 \
    trainer.total_episodes=20 \
    trainer.nnodes=4 \
    trainer.n_gpus_per_node=8 \
    worker.rollout.n=32 \
    worker.actor.global_batch_size=512
```

## Configuration via YAML

Create a custom configuration file:

```yaml
# custom_curriculum.yaml
defaults:
  - mmr1_b200

data:
  sampling_strategy: curriculum
  curriculum_metrics:
    - learnability
    - self-bleu
  curriculum_metric_weights:
    - 0.7
    - 0.3
  curriculum_mixture_ratio: 0.6
  curriculum_update_freq: 10
  curriculum_rollout_n: 12
```

Use with:

```bash
python -m verl.trainer.main config=custom_curriculum.yaml
```

## Mixture Ratio Guidelines

The `curriculum_mixture_ratio` controls the balance between weighted and random sampling:

| Ratio | Effect | Use Case |
|-------|--------|----------|
| 0.0 | Fully random | Equivalent to shuffle strategy |
| 0.2 | 20% weighted, 80% random | Light curriculum influence |
| 0.5 | Balanced | Good starting point |
| 0.8 | 80% weighted, 20% random | Strong curriculum focus |
| 1.0 | Fully weighted | Pure curriculum learning |

## Update Frequency Strategies

| Setting | Behavior | Best For |
|---------|----------|----------|
| 0 | Epoch-level updates | Small datasets |
| 4 | Every 4 steps | Balanced approach |
| 8-10 | Less frequent | Large datasets |
| 20+ | Infrequent | Stable training |

## Performance Tips

### For Small Datasets (<10K samples)
- Use `curriculum_update_freq=0` for epoch-level updates
- Lower `curriculum_rollout_n` to 4-8
- Consider higher `curriculum_momentum` (0.5-0.9)

### For Large Models (>30B parameters)
- Reduce `curriculum_rollout_batch_size` to manage memory
- Use fewer curriculum metrics to reduce computation
- Consider `curriculum_update_freq=20+` for stability

### For Quick Experimentation
- Start with single metric: `'data.curriculum_metrics=[learnability]'`
- Use `curriculum_mixture_ratio=0.5` as baseline
- Set `curriculum_rollout_n=4` for faster iteration

## Monitoring Curriculum Learning

Track these metrics in wandb/tensorboard:
- `curriculum/mean_weight`: Average sample weight
- `curriculum/std_weight`: Weight distribution spread
- `curriculum/min_weight`, `curriculum/max_weight`: Weight range
- `curriculum/consumed_batches`: Training progress
- `curriculum/random_position`: Mix of weighted vs random samples

## Resume from Checkpoint

Curriculum weights and sampler state are automatically saved:

```bash
trainer.load_checkpoint_path=/checkpoints/previous_run/global_step_100 \
trainer.save_checkpoint_path=/checkpoints/continued_run
```

## Troubleshooting

### Issue: Weights not updating
- Check `curriculum_update_freq` is set appropriately
- Verify metrics are being calculated (check logs)
- Ensure `curriculum_rollout_batch_size` is reasonable

### Issue: Training instability
- Reduce `curriculum_mixture_ratio` for more randomness
- Increase `curriculum_momentum` for smoother updates
- Use fewer or simpler metrics

### Issue: Slow metric computation
- Reduce `curriculum_rollout_n`
- Increase `curriculum_rollout_batch_size`
- Use fewer metrics or simpler metrics (e.g., just learnability)

## Citation

If you use curriculum learning in EasyR1, please cite:

```bibtex
@software{easyr1_curriculum,
  title={EasyR1: Scalable RLHF with Curriculum Learning},
  year={2024},
  publisher={ByteDance},
}
```