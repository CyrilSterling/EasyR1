# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Reward config
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class RewardConfig:
    reward_type: str = "function"
    compute_score: str = "math"
    batch_processing: bool = False
    cos_len_reward_config: Dict[str, Any] = field(default_factory=dict)
