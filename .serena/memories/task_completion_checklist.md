# EasyR1 Task Completion Checklist

## When a Development Task is Completed

### 1. Code Quality Checks (REQUIRED)
Run these commands before considering any task complete:

```bash
# Check code quality and formatting
make quality

# Auto-fix any formatting issues
make style
```

Or run individual commands:
```bash
# Check for issues
ruff check verl scripts setup.py

# Format code  
ruff format verl scripts setup.py

# Fix issues automatically
ruff check verl scripts setup.py --fix
```

### 2. Pre-commit Validation
```bash
# Run all pre-commit hooks
make commit

# Or run pre-commit manually
pre-commit run --all-files
```

### 3. Testing (if applicable)
```bash
# Run available tests (limited test coverage)
python3 -m pytest verl/trainer/EasyR1/verl/trainer/test_ray_logic.py

# For training-related changes, consider running a quick debug training
bash examples/debug.sh
```

### 4. Installation Verification
```bash
# Verify package can still be installed
pip install -e .
```

### 5. Configuration Validation (for training changes)
```bash
# Validate YAML configs can be parsed
python3 -c "import yaml; yaml.safe_load(open('examples/config.yaml'))"
```

## Specific Task Types

### For Model/Training Changes
- [ ] Run code quality checks (`make quality`, `make style`)
- [ ] Test with debug configuration (`bash examples/debug.sh`)
- [ ] Verify config files are still valid
- [ ] Check that imports work correctly

### For Documentation Changes
- [ ] Check markdown syntax
- [ ] Verify links work
- [ ] Run pre-commit hooks (`make commit`)

### For Configuration Changes
- [ ] Validate YAML syntax
- [ ] Test with a simple training run
- [ ] Ensure backward compatibility

### For Infrastructure Changes
- [ ] Test package installation (`pip install -e .`)
- [ ] Verify Docker builds still work (if applicable)
- [ ] Run full pre-commit suite

## Git Workflow
1. Make your changes
2. Run task completion checklist
3. Stage changes: `git add .`
4. Commit with descriptive message: `git commit -m "description"`
5. Push to branch: `git push`

## Notes
- The project has minimal automated testing, so manual verification is crucial
- Always run `make quality` and `make style` before committing
- Pre-commit hooks will catch many issues automatically
- For training-related changes, testing with debug configs is recommended
- The project uses Apache 2.0 license, maintain license headers where applicable