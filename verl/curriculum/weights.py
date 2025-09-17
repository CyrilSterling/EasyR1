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
Curriculum weight management for sample weighting strategies.
"""

import os
from typing import Optional, Dict, Any
import torch
import ray
from torchdata.stateful_dataloader import StatefulDataLoader

from ..protocol import DataProto
from ..utils.dataset import collate_fn
from .samplers import PaddedSequentialSampler
from .metrics import combine_metric_results


class CurriculumWeightManager:
    """Manages curriculum weights computation, saving, and loading."""

    def __init__(self, config, train_dataset, actor_rollout_wg, reward_fn):
        """Initialize the curriculum weight manager.

        Args:
            config: Training configuration
            train_dataset: Training dataset
            actor_rollout_wg: Actor rollout worker group for generation
            reward_fn: Reward function for metric calculation
        """
        self.config = config
        self.train_dataset = train_dataset
        self.actor_rollout_wg = actor_rollout_wg
        self.reward_fn = reward_fn

    def save_weights(self, weights: torch.Tensor, step: Optional[int] = None) -> None:
        """Save curriculum weights to a file."""
        # Create directory if it doesn't exist
        weights_dir = os.path.join(
            self.config.trainer.save_checkpoint_path, "curriculum_weights"
        )
        os.makedirs(weights_dir, exist_ok=True)

        # Create filename based on step (if provided) or use 'initial'
        filename = (
            f"weights_step_{step}.pt" if step is not None else "initial_weights.pt"
        )
        weights_path = os.path.join(weights_dir, filename)

        # Save weights
        torch.save(weights, weights_path)

    def compute_weights(self) -> torch.Tensor:
        """Compute curriculum weights for each sample in the dataset using dataloader for efficient loading."""

        curriculum_weights = torch.zeros(len(self.train_dataset), dtype=torch.float32)

        # Put reward_fn into Ray object store once
        reward_fn_ref = ray.put(self.reward_fn)

        # Create a temporary dataloader with padded sequential sampling
        padded_sampler = PaddedSequentialSampler(
            dataset=self.train_dataset,
            batch_size=self.config.data.curriculum_rollout_batch_size
        )

        curriculum_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.curriculum_rollout_batch_size,
            sampler=padded_sampler,
            num_workers=8,  # Same as train_dataloader for prefetch
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,  # PaddedSequentialSampler handles the padding
        )

        # Process batches sequentially but don't wait for metric calculations
        metric_futures_with_batch_idx = []  # Store (batch_idx, futures_dict) tuples

        for batch_idx, batch_data in enumerate(curriculum_dataloader):
            # Calculate metrics using pre-loaded batch_data
            metric_futures_cur_batch = self._calculate_curriculum_metric_with_batch(
                batch_data=batch_data,
                reward_fn_ref=reward_fn_ref  # Pass the ObjectRef instead of the object
            )

            # Store batch index and futures for later collection
            metric_futures_with_batch_idx.append((batch_idx, metric_futures_cur_batch))

        # Now process all the pending futures
        for batch_idx, futures in metric_futures_with_batch_idx:
            # Collect results from futures
            metric_results = ray.get(futures)

            # Combine the metric results for this batch
            combined_metric = combine_metric_results(
                metric_results,
                weights=self.config.data.curriculum_metric_weights,
                metrics=self.config.data.curriculum_metrics,
            )

            # Store the combined metric, handling potential padding
            start_idx = batch_idx * self.config.data.curriculum_rollout_batch_size
            end_idx = min(
                start_idx + self.config.data.curriculum_rollout_batch_size,
                len(self.train_dataset),
            )

            # Only assign weights to valid (non-padded) samples
            valid_samples = end_idx - start_idx
            curriculum_weights[start_idx:end_idx] = combined_metric.detach()[:valid_samples]

        # Normalize weights using min-max scaling
        min_weight = curriculum_weights.min()
        max_weight = curriculum_weights.max()
        curriculum_weights = (curriculum_weights - min_weight) / (
            max_weight
            - min_weight
            + 1e-8  # Add small epsilon to avoid division by zero
        )

        return curriculum_weights

    def update_weights(self, current_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Update curriculum learning weights based on current model performance."""
        # Create a temporary dataloader for weight estimation
        new_weights = self.compute_weights()

        # Update weights with configured momentum
        momentum = self.config.data.curriculum_momentum

        if current_weights is None:
            return new_weights
        else:
            return momentum * current_weights + (1 - momentum) * new_weights

    def _calculate_curriculum_metric_with_batch(
        self, batch_data: Dict, reward_fn_ref: ray.ObjectRef
    ) -> list[ray.ObjectRef]:
        """Calculate curriculum metrics using pre-loaded batch data from dataloader."""

        # Prepare gen batch
        batch = DataProto.from_single_dict(batch_data)
        batch_size = len(batch.batch["input_ids"])

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
                from .metrics import calculate_learnability_metric_with_batch_data
                future = calculate_learnability_metric_with_batch_data.remote(
                    reward_fn_ref,  # Pass the ObjectRef
                    batch_data,  # Pass the pre-extracted batch_data dictionary
                    gen_batch_output,
                    batch_size=batch_size,
                    curriculum_rollout_n=self.config.data.curriculum_rollout_n,
                )
            elif metric_name == "distinct":
                from .metrics import calculate_distinct_n_metric
                future = calculate_distinct_n_metric.remote(
                    gen_batch_output.batch["responses"],
                    batch_size=batch_size,
                    curriculum_rollout_n=self.config.data.curriculum_rollout_n,
                )
            elif metric_name == "self-bleu":
                from .metrics import calculate_self_bleu_metric
                future = calculate_self_bleu_metric.remote(
                    gen_batch_output.batch["responses"],
                    batch_size=batch_size,
                    curriculum_rollout_n=self.config.data.curriculum_rollout_n,
                )
            elif metric_name == "edit-distance":
                from .metrics import calculate_edit_distance_metric
                future = calculate_edit_distance_metric.remote(
                    gen_batch_output.batch["responses"],
                    batch_size=batch_size,
                    curriculum_rollout_n=self.config.data.curriculum_rollout_n,
                )
            else:
                raise ValueError(f"Unknown curriculum metric: {metric_name}")

            metric_futures.append(future)

        return metric_futures

    def load_curriculum_state(self, checkpoint_path: Optional[str]) -> Optional[Dict[str, Any]]:
        """Load curriculum state from checkpoint."""
        if checkpoint_path is None:
            return None

        curriculum_path = os.path.join(checkpoint_path, "curriculum_state.pt")
        if os.path.exists(curriculum_path):
            curriculum_state = torch.load(curriculum_path, weights_only=False)
            return curriculum_state
        else:
            return None

    def save_curriculum_state(self, folder_path: str, curriculum_weights: torch.Tensor, sampler_state: Dict) -> None:
        """Save curriculum state to checkpoint."""
        curriculum_state = {
            "curriculum_weights": curriculum_weights,
            "sampler_state": sampler_state,
        }
        curriculum_path = os.path.join(folder_path, "curriculum_state.pt")
        torch.save(curriculum_state, curriculum_path)