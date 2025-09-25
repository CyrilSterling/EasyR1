# EasyR1 Code Style and Conventions

## Code Formatting
The project uses **Ruff** as the primary linting and formatting tool.

### Ruff Configuration (from pyproject.toml)
```toml
[tool.ruff]
target-version = "py39"
line-length = 119
indent-width = 4

[tool.ruff.lint]
ignore = ["C901", "E501", "E741", "W605", "C408"]
select = ["C", "E", "F", "I", "W", "RUF022"]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["E402", "F401", "F403", "F811"]

[tool.ruff.lint.isort]
lines-after-imports = 2
known-first-party = ["verl"]
known-third-party = ["torch", "transformers", "wandb"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"
```

## Python Style Guidelines

### General Conventions
- **Python Version**: 3.9+ (project uses 3.11.13)
- **Line Length**: 119 characters maximum
- **Indentation**: 4 spaces (no tabs)
- **Quote Style**: Double quotes preferred
- **Import Organization**: Follow isort configuration with 2 lines after imports

### Import Structure
1. Standard library imports
2. Third-party imports (torch, transformers, wandb)
3. First-party imports (verl)
4. Two blank lines after imports

Example from codebase:
```python
from typing import Optional, Tuple

import torch

from .flash_attention_utils import flash_attention_forward


try:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
        Qwen2VLAttention,
        apply_multimodal_rotary_pos_emb,
        repeat_kv,
    )
except ImportError:
    pass
```

### Documentation Style
- Use docstrings for functions and classes
- Copyright headers in files when applicable
- Apache 2.0 license headers for significant files

Example copyright header:
```python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
```

### Naming Conventions
- **Variables/Functions**: snake_case
- **Classes**: PascalCase
- **Constants**: UPPER_SNAKE_CASE
- **Module Names**: lowercase with underscores

### Type Hints
- Use type hints for function parameters and return values
- Import from `typing` module when needed
- Example: `Optional[torch.Tensor] = None`

## File Organization
- **Init files**: Can ignore certain lint rules (E402, F401, F403, F811)
- **Module structure**: Follow the pattern seen in verl/ directory
- **Configuration files**: YAML format for training configs

## Pre-commit Hooks
The project uses pre-commit hooks for code quality:
- check-ast: Validate Python syntax
- check-added-large-files: Prevent large file commits (25MB limit)
- check-merge-conflict: Detect merge conflicts
- check-yaml: Validate YAML files
- debug-statements: Remove debug statements
- end-of-file-fixer: Ensure files end with newline
- trailing-whitespace: Remove trailing whitespace
- pyupgrade: Upgrade Python syntax to 3.8+

## Ignored Lint Rules
- C901: Complex structure (function complexity)
- E501: Line too long (handled by formatter)
- E741: Ambiguous variable name
- W605: Invalid escape sequence
- C408: Unnecessary dict call