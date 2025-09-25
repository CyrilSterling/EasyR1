# EasyR1 Project Overview

## Purpose
EasyR1 is an efficient, scalable, multi-modality RL (Reinforcement Learning) training framework. It's a clean fork of the original veRL project, specifically enhanced to support vision language models (VLMs). The framework is designed for training large language models using reinforcement learning techniques.

## Key Features
- Support for both language models (Llama3/Qwen2/Qwen2.5) and vision language models (Qwen2/Qwen2.5-VL)
- Multiple RL algorithms: GRPO, Reinforce++, ReMax, RLOO
- Efficient training with HybridEngine and vLLM's SPMD mode
- Support for multi-modal datasets (text and vision-text)
- Advanced features: padding-free training, checkpoint resuming, multiple logging backends

## Tech Stack
- **Language**: Python 3.9+ (project uses 3.11.13)
- **Core ML Libraries**: 
  - transformers==4.51.2
  - torch==2.6.0
  - vllm==0.8.4
- **RL Framework**: Built on top of Ray for distributed training
- **Additional Libraries**: accelerate, datasets, peft, pillow, wandb, tensorboard
- **Development Tools**: ruff for linting/formatting, pre-commit hooks

## Main Directories Structure
```
EasyR1/
├── verl/                    # Core RL training framework
│   ├── trainer/            # Training logic and main entry points
│   ├── workers/            # Distributed worker components
│   ├── models/             # Model-specific implementations
│   ├── single_controller/  # Controller components
│   └── utils/              # Utility functions
├── examples/               # Training example scripts and configs
│   ├── baselines/          # Baseline training scripts
│   └── mmr1/              # MMR1-specific examples
├── docs/                   # Documentation
├── quick_start/            # Quick start scripts
└── assets/                 # Images and assets
```

## Installation
The project is installed as a Python package using:
```bash
pip install -e .
```

## Main Use Cases
1. Training vision-language models on mathematical reasoning tasks
2. Running GRPO (Generalized Reward-based Policy Optimization) training
3. Multi-modal RL training with custom datasets
4. Distributed training across multiple GPUs/nodes