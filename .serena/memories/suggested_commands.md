# EasyR1 Suggested Commands

## Development Commands

### Linting and Formatting
```bash
# Check code quality and formatting
make quality

# Auto-fix linting and formatting issues  
make style

# Run individual ruff commands
ruff check verl scripts setup.py
ruff format verl scripts setup.py

# Fix issues automatically
ruff check verl scripts setup.py --fix
ruff format verl scripts setup.py
```

### Pre-commit Hooks
```bash
# Install and run pre-commit hooks
make commit

# Install pre-commit hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

### Build
```bash
# Build the package
make build
python3 setup.py sdist bdist_wheel
```

## Training Commands

### Main Training Entry Point
```bash
# Run RL training with config
python3 -m verl.trainer.main config=examples/config.yaml [additional_args...]
```

### Example Training Scripts
```bash
# Qwen2.5-VL 7B on Geometry3K dataset with GRPO
bash examples/qwen2_5_vl_7b_geo3k_grpo.sh

# Qwen2.5-VL 3B on Geometry3K dataset  
bash examples/qwen2_5_vl_3b_geo3k_grpo.sh

# Mathematical reasoning with text-only model
bash examples/qwen2_5_7b_math_grpo.sh

# Debug script
bash examples/debug.sh
```

### Model Merging
```bash
# Merge checkpoint to Hugging Face format
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## System Utilities

### File Operations
```bash
# Use fd instead of find (faster)
fd PATTERN                    # Find files/dirs matching pattern
fd -t f "*.py"               # Find Python files
fd -t d                      # Find directories only

# Use ag instead of grep (faster) 
ag PATTERN                   # Search for pattern in files
```

### Git Operations
```bash
git status
git add .
git commit -m "message"
git push
```

### Environment
```bash
# Python version check
python3 --version

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Docker (recommended)
docker pull hiyouga/verl:ngc-th2.5.1-cu120-vllm0.7.4-hotfix
```

## Testing

### Run Tests
```bash
# Currently limited test files available
python3 -m pytest verl/trainer/EasyR1/verl/trainer/test_ray_logic.py
```

Note: The project has minimal test coverage with only one test file found at `verl/trainer/EasyR1/verl/trainer/test_ray_logic.py`.