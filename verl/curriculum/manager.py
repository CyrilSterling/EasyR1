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
Curriculum learning manager for coordinating all curriculum components.
"""

import os
from typing import Optional, Dict, Any, List
import torch
import ray

from ..protocol import DataProto
from ..utils.dataset import collate_fn
from .samplers import MixedCurriculumSampler
from .weights import CurriculumWeightManager
from .metrics import (
    calculate_learnability_metric_with_batch_data,
    calculate_distinct_n_metric,
    calculate_self_bleu_metric,
    calculate_edit_distance_metric,
    combine_metric_results,
)


class CurriculumManager:
    """Manages all aspects of curriculum learning including weights, sampling, and metrics."""

    def __init__(
        self,
        config,
        train_dataset,
        actor_rollout_wg,
        reward_fn,
        checkpoint_path: Optional[str] = None,
    ):
        """Initialize the curriculum manager.

        Args:
            config: Training configuration
            train_dataset: Training dataset
            actor_rollout_wg: Actor rollout worker group for generation
            reward_fn: Reward function for metric calculation
            checkpoint_path: Optional checkpoint path for loading saved state
        """
        self.config = config
        self.train_dataset = train_dataset
        self.actor_rollout_wg = actor_rollout_wg
        self.reward_fn = reward_fn
        self.checkpoint_path = checkpoint_path

        # Initialize weight manager
        self.weight_manager = CurriculumWeightManager(
            config, train_dataset, actor_rollout_wg, reward_fn
        )

        # Load or compute initial weights
        self._initialize_weights()

        # Initialize curriculum sampler
        self.sampler = MixedCurriculumSampler(
            dataset=train_dataset,
            weights=self.train_dataset.curriculum_weights,
            batch_size=config.data.rollout_batch_size,
            mixture_ratio=config.data.curriculum_mixture_ratio,
            replacement=True,  # Always use replacement for weighted sampling
            generator=torch.Generator().manual_seed(config.data.seed),
        )

        # Load sampler state if available
        if self.checkpoint_path:
            self._load_sampler_state()

        # Update sampler with current weights
        self.sampler.update_weights(self.train_dataset.curriculum_weights)

    def _initialize_weights(self):
        """Initialize curriculum weights from checkpoint or compute new ones."""
        curriculum_state = self.weight_manager.load_curriculum_state(self.checkpoint_path)

        if curriculum_state is not None:
            self.train_dataset.curriculum_weights = curriculum_state["curriculum_weights"]
        else:
            # Compute initial weights
            self.update_weights()

    def _load_sampler_state(self):
        """Load sampler state from checkpoint if available."""
        curriculum_state = self.weight_manager.load_curriculum_state(self.checkpoint_path)
        if curriculum_state is not None and "sampler_state" in curriculum_state:
            self.sampler.load_state_dict(curriculum_state["sampler_state"])

    def update_weights(self, global_step: Optional[int] = None):
        """Update curriculum weights based on current model performance."""
        # Get current weights
        current_weights = getattr(self.train_dataset, "curriculum_weights", None)

        # Update weights with momentum
        new_weights = self.weight_manager.update_weights(current_weights)
        self.train_dataset.curriculum_weights = new_weights

        # Save updated weights
        if global_step is not None:
            self.weight_manager.save_weights(new_weights, step=global_step)

        # Update sampler with new weights
        self.sampler.update_weights(new_weights)

    def get_sampler(self):
        """Get the curriculum sampler."""
        return self.sampler

    def reset_cached_indices(self):
        """Reset cached indices in the sampler for new epoch."""
        self.sampler.cached_mixed_indices = None

    def update_position_after_batch(self):
        """Update sampler position after processing a batch."""
        self.sampler.update_position_after_batch()

    def get_metrics(self) -> Dict[str, float]:
        """Get current curriculum metrics for logging."""
        if not hasattr(self.train_dataset, "curriculum_weights"):
            return {}

        weights = self.train_dataset.curriculum_weights
        return {
            "curriculum/mean_weight": weights.mean().item(),
            "curriculum/std_weight": weights.std().item(),
            "curriculum/min_weight": weights.min().item(),
            "curriculum/max_weight": weights.max().item(),
            "curriculum/consumed_batches": self.sampler.consumed_batches,
            "curriculum/random_position": self.sampler.get_training_random_position(),
        }

    def save_state(self, folder_path: str):
        """Save curriculum state to checkpoint."""
        if hasattr(self.train_dataset, "curriculum_weights"):
            self.weight_manager.save_curriculum_state(
                folder_path,
                self.train_dataset.curriculum_weights,
                self.sampler.state_dict(),
            )

    def calculate_curriculum_metric(
        self, dataset, indices: List[int]
    ) -> list[ray.ObjectRef]:
        """Calculate the curriculum metric based on the configured strategy.

        Returns:
            A list containing futures for the remote metric calculations.
        """

        # Prepare gen batch
        batch_data = collate_fn([dataset[i] for i in indices])
        batch = DataProto.from_single_dict(batch_data)
        batch_size = len(indices)

        # Generate responses for the batch
        if "multi_modal_inputs" in batch.non_tensor_batch.keys():
            gen_batch = batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=[
                    "raw_prompt_ids",
                    "multi_modal_data",
                    "multi_modal_inputs",
                ],
            )
        else:
            gen_batch = batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids"],
            )

        # Generate responses
        gen_batch.meta_info["n"] = self.config.data.curriculum_rollout_n
        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
        gen_batch.meta_info.pop("n")

        # Submit futures for remote metric calculations
        metric_futures = []

        for metric_name in self.config.data.curriculum_metrics:
            if metric_name == "learnability":
                future = calculate_learnability_metric_with_batch_data.remote(
                    self.reward_fn,
                    batch_data,
                    gen_batch_output,
                    batch_size=batch_size,
                    curriculum_rollout_n=self.config.data.curriculum_rollout_n,
                )
            elif metric_name == "distinct":
                future = calculate_distinct_n_metric.remote(
                    gen_batch_output.batch["responses"],
                    batch_size=batch_size,
                    curriculum_rollout_n=self.config.data.curriculum_rollout_n,
                )
            elif metric_name == "self-bleu":
                future = calculate_self_bleu_metric.remote(
                    gen_batch_output.batch["responses"],
                    batch_size=batch_size,
                    curriculum_rollout_n=self.config.data.curriculum_rollout_n,
                )
            elif metric_name == "edit-distance":
                future = calculate_edit_distance_metric.remote(
                    gen_batch_output.batch["responses"],
                    batch_size=batch_size,
                    curriculum_rollout_n=self.config.data.curriculum_rollout_n,
                )
            else:
                raise ValueError(f"Unknown curriculum metric: {metric_name}")

            metric_futures.append(future)

        return metric_futures

    def should_update_weights(self, global_step: int) -> bool:
        """Check if weights should be updated at the current step."""
        if self.config.data.curriculum_update_freq <= 0:
            return False  # Epoch-level updates
        return global_step % self.config.data.curriculum_update_freq == 0

    def should_update_weights_epoch(self) -> bool:
        """Check if weights should be updated at epoch level."""
        return self.config.data.curriculum_update_freq == 0