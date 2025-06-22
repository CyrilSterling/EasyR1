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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import multiprocessing
import os
import os.path as osp
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Type, Tuple

import numpy as np
import ray
import torch
from codetiming import Timer
from Levenshtein import distance
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from PIL import Image
from ray.experimental.tqdm_ray import tqdm
from torch.utils.data import RandomSampler, SequentialSampler, WeightedRandomSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
from ..utils.dataset import RLHFDataset, collate_fn
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str
from ..utils.seqlen_balancing import (
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)
from ..workers.fsdp_workers import FSDPWorker
from . import core_algos
from .config import PPOConfig
from .metrics import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)


# Allow very large images
Image.MAX_IMAGE_PIXELS = None


WorkerType = Type[Worker]


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix=resource_pool_name,
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum(
            [
                n_gpus
                for process_on_nodes in self.resource_pool_spec.values()
                for n_gpus in process_on_nodes
            ]
        )

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [
                n_gpus
                for process_on_nodes in self.resource_pool_spec.values()
                for n_gpus in process_on_nodes
            ]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}."
            )


def apply_kl_penalty(
    data: DataProto, kl_ctrl: core_algos.KLController, kl_penalty="kl"
):
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    if "ref_log_probs" in data.batch.keys():
        kld = core_algos.kl_penalty(
            data.batch["old_log_probs"],
            data.batch["ref_log_probs"],
            kl_penalty=kl_penalty,
        )  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = VF.masked_mean(
        kld, mask=response_mask, dim=-1
    )  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"critic/kl": current_kl, "critic/kl_coef": beta}
    return data, metrics


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
):
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            eos_mask=response_mask,
            gamma=gamma,
            lam=lam,
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, index=index
        )
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma
        )
    elif adv_estimator == AdvantageEstimator.REMAX:
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=token_level_rewards,
            reward_baselines=reward_baselines,
            eos_mask=response_mask,
        )
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, index=index
        )
    else:
        raise NotImplementedError

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield

    timing_raw[name] = timer.last


class CurriculumWeightedSampler(WeightedRandomSampler):
    """Custom weighted sampler that supports dynamic weight updates."""

    def __init__(self, weights, num_samples, replacement=True, generator=None):
        super().__init__(weights, num_samples, replacement, generator)
        self.weights = weights

    def update_weights(self, new_weights):
        """Update the sampling weights."""
        self.weights = new_weights


class MixedCurriculumSampler(torch.utils.data.Sampler):
    """Custom sampler that mixes random and weighted sampling."""

    def __init__(
        self,
        dataset,
        weights,
        batch_size,
        mixture_ratio=0.5,
        replacement=True,
        generator=None,
    ):
        self.dataset = dataset
        self.weights = weights
        self.batch_size = batch_size
        self.mixture_ratio = mixture_ratio
        self.replacement = replacement
        self.generator = generator
        self.dataset_size = len(dataset)

        # Calculate number of samples of each type per batch
        self.n_weighted_per_batch = int(batch_size * mixture_ratio)
        self.n_random_per_batch = batch_size - self.n_weighted_per_batch

        # Track total number of random indices needed for the entire dataset
        self.total_random_indices_needed = (
            self.dataset_size // self.batch_size
        ) * self.n_random_per_batch
        if self.dataset_size % self.batch_size > 0:
            self.total_random_indices_needed += self.dataset_size % self.batch_size

        # Create weighted sampler for the entire dataset
        self.weighted_sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=self.dataset_size,  # Sample as many as the dataset size
            replacement=replacement,
            generator=generator,
        )

        # Initialize random indices pool and tracking
        self._init_random_indices()

        # Cache for mixed indices to handle multiple calls to __iter__
        self.cached_mixed_indices = None

        # Separate tracking variable for actual consumed batches in training
        self.consumed_batches = 0

    def _init_random_indices(self):
        """Initialize random indices - creating a pool of random indices for the dataset."""
        # Create a random permutation of all indices
        if self.generator is not None:
            rand_indices = torch.randperm(
                len(self.dataset), generator=self.generator
            ).tolist()
        else:
            import random

            rand_indices = list(range(len(self.dataset)))
            random.shuffle(rand_indices)

        # Store the entire random permutation
        self.random_indices_pool = rand_indices
        # Position tracker for random indices used for initial index creation
        self.random_indices_position = 0

    def _get_next_random_indices(self, n):
        """Get next n random indices from our pre-shuffled pool, cycling if needed."""
        # If we're near the end of our pool, we need to cycle back
        if self.random_indices_position + n > len(self.random_indices_pool):
            # Just wrap around to the beginning without reshuffling
            self.random_indices_position = 0

        # Get the indices
        indices = self.random_indices_pool[
            self.random_indices_position : self.random_indices_position + n
        ]
        self.random_indices_position += n
        return indices

    def get_training_random_position(self):
        """Get the current position in random indices used for actual training.
        This is separate from the position used during index generation.
        """
        # Simply use modulo to handle wrap-around consistently with _get_next_random_indices
        return (self.consumed_batches * self.n_random_per_batch) % len(
            self.random_indices_pool
        )

    def update_position_after_batch(self):
        """Update tracking of consumed batches after each batch is processed in training.
        This should be called after each batch is processed in the training loop.
        """
        # Increment the number of consumed batches
        self.consumed_batches += 1

    def __iter__(self):
        # Only compute indices if they haven't been cached
        if self.cached_mixed_indices is None:
            # Get all indices from weighted sampler
            weighted_indices = list(self.weighted_sampler)

            # Shuffle the weighted indices
            if self.generator is not None:
                rand_tensor = torch.randperm(
                    len(weighted_indices), generator=self.generator
                )
                weighted_indices = [weighted_indices[i] for i in rand_tensor]
            else:
                import random

                random.shuffle(weighted_indices)

            # Create batches that maintain the mixture ratio
            mixed_indices = []
            num_batches = self.dataset_size // self.batch_size

            # Synchronize the position counter with the actual training position
            # This ensures we continue from where training left off rather than restarting
            self.random_indices_position = self.get_training_random_position()
            print(
                f"Starting index generation with random_indices_position: {self.random_indices_position}"
            )

            for batch_idx in range(num_batches):
                # Calculate start indices for weighted part
                w_start = batch_idx * self.n_weighted_per_batch

                # Get indices for this batch
                batch_weighted = weighted_indices[
                    w_start : w_start + self.n_weighted_per_batch
                ]
                # Get random indices from our tracked pool
                batch_random = self._get_next_random_indices(self.n_random_per_batch)

                # Interleave the indices to maintain diversity within the batch
                batch_indices = []
                for i in range(max(len(batch_weighted), len(batch_random))):
                    if i < len(batch_weighted):
                        batch_indices.append(batch_weighted[i])
                    if i < len(batch_random):
                        batch_indices.append(batch_random[i])

                mixed_indices.extend(batch_indices)

            # Handle remaining samples if dataset size is not divisible by batch size
            remaining = self.dataset_size % self.batch_size
            if remaining > 0:
                w_start = num_batches * self.n_weighted_per_batch

                n_remaining_weighted = int(remaining * self.mixture_ratio)
                n_remaining_random = remaining - n_remaining_weighted

                batch_weighted = weighted_indices[
                    w_start : w_start + n_remaining_weighted
                ]
                batch_random = self._get_next_random_indices(n_remaining_random)

                batch_indices = []
                for i in range(max(len(batch_weighted), len(batch_random))):
                    if i < len(batch_weighted):
                        batch_indices.append(batch_weighted[i])
                    if i < len(batch_random):
                        batch_indices.append(batch_random[i])

                mixed_indices.extend(batch_indices)

            # Store the computed indices
            self.cached_mixed_indices = mixed_indices

        # Return iterator of cached indices
        return iter(self.cached_mixed_indices)

    def __len__(self):
        return self.dataset_size

    def update_weights(self, new_weights):
        """Update the sampling weights."""
        self.weights = new_weights
        self.weighted_sampler = WeightedRandomSampler(
            weights=new_weights,
            num_samples=self.dataset_size,
            replacement=self.replacement,
            generator=self.generator,
        )
        # Reset the cached indices to force regeneration with new weights
        self.cached_mixed_indices = None
        # Note: We don't reset random indices - this preserves their state between updates


class PaddedSequentialSampler(torch.utils.data.BatchSampler):
    """Custom sampler that pads each batch to the same batch size."""

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        pad_to_length = (len(dataset) + batch_size - 1) // batch_size * batch_size
        self.indices = list(range(pad_to_length))
        self.indices[len(dataset) :] = [len(dataset) - 1] * (
            pad_to_length - len(dataset)
        )

    def __iter__(self):
        """Generate padded batches of the same size."""
        for i in self.indices:
            yield i

    def __len__(self):
        return len(self.indices)


@ray.remote
def calculate_learnability_metric(reward_fn, batch, batch_size, curriculum_rollout_n):
    """Remote task for calculating learnability metric."""
    reward_tensor, reward_metrics = reward_fn(batch)
    # reshape to (batch_size, n, max_len)
    acc_reward_tensor = torch.tensor(
        reward_metrics["accuracy"], dtype=torch.float32
    ).view(batch_size, curriculum_rollout_n, -1)

    # Calculate pass rate and learnability
    pass_rate = acc_reward_tensor.mean(dim=-1).mean(dim=-1)  # sequence-level average over rollouts
    learnability = pass_rate * (1 - pass_rate)
    return learnability


@ray.remote
def calculate_distinct_n_metric(responses, batch_size, curriculum_rollout_n, n=3):
    """Remote task for calculating distinct-n metric."""
    tokenized_sequences = responses.tolist()
    distinct_score = []
    for i in range(batch_size):
        distinct_score.append(
            calculate_distinct_n(
                tokenized_sequences[
                    i * curriculum_rollout_n : (i + 1) * curriculum_rollout_n
                ],
                n=n,
            )
        )
    return torch.tensor(distinct_score, dtype=torch.float32)


@ray.remote
def calculate_self_bleu_metric(responses, batch_size, curriculum_rollout_n):
    """Remote task for calculating self-BLEU-123 metric."""
    tokenized_sequences = responses.tolist()
    
    # Create a partial function with fixed arguments
    worker_fn = partial(
        calculate_batch_self_bleu_123_helper,
        tokens=tokenized_sequences,
        n_per_item=curriculum_rollout_n,
    )

    # Use multiprocessing to compute BLEU scores in parallel
    with multiprocessing.Pool(processes=32) as pool:
        bleu_score = pool.map(worker_fn, range(batch_size))
    
    return torch.tensor(bleu_score, dtype=torch.float32)


@ray.remote
def calculate_edit_distance_metric(responses, batch_size, curriculum_rollout_n):
    """Remote task for calculating edit distance metric."""
    tokenized_sequences = responses.tolist()
    
    worker_fn = partial(
        calculate_pairwise_edit_distance_helper,
        tokens=tokenized_sequences,
        n_per_item=curriculum_rollout_n,
    )
    
    with multiprocessing.Pool(processes=32) as pool:
        edit_score = pool.map(worker_fn, range(batch_size))
    
    return torch.tensor(edit_score, dtype=torch.float32)


def combine_metric_results(results, weights, metrics):
    """Combine metric results with their weights."""
    weighted_sum = None
    for i, (result, metric) in enumerate(zip(results, metrics)):
        weight = weights[i]
        if weighted_sum is None:
            weighted_sum = result * weight
        else:
            weighted_sum += result * weight
    return weighted_sum


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn: Callable = None,
        val_reward_fn: Callable = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.worker.hybrid_engine
        if self.hybrid_engine:
            assert (
                Role.ActorRollout in role_worker_mapping
            ), f"ActorRollout should be included in {role_worker_mapping.keys()}."
        else:
            raise NotImplementedError

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if Role.RefPolicy in role_worker_mapping and not config.algorithm.disable_kl:
            self.use_reference_policy = True
            self.kl_ctrl = core_algos.get_kl_controller(config.algorithm)
        else:
            self.use_reference_policy = False
            self.kl_ctrl = core_algos.FixedKLController(init_kl_coef=0.0)
            print(
                "KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics."
            )

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(
                f"Unknown advantage estimator: {config.algorithm.adv_estimator}."
            )

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError(
                "Rollout batch size must be divisible by global batch size."
            )

        if (
            self.use_critic
            and config.data.rollout_batch_size % config.worker.critic.global_batch_size
            != 0
        ):
            raise ValueError(
                "Rollout batch size must be divisible by global batch size."
            )

        # Initialize workers first
        self.init_workers()

        # Then create dataloader which depends on workers
        self._create_dataloader()

    def _create_dataloader(self) -> None:
        self.train_dataset = RLHFDataset(
            data_path=self.config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            answer_key=self.config.data.answer_key,
            image_key=self.config.data.image_key,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation="right",
            system_prompt=self.config.data.system_prompt,
            min_pixels=self.config.data.min_pixels,
            max_pixels=self.config.data.max_pixels,
        )

        # breakpoint()
        # Initialize sampler based on configured strategy
        if self.config.data.sampling_strategy == "curriculum":
            # Initialize curriculum learning weights
            self._init_curriculum_weights()

            # Create mixed curriculum sampler
            self.curriculum_sampler = MixedCurriculumSampler(
                dataset=self.train_dataset,
                weights=self.curriculum_weights,
                batch_size=self.config.data.rollout_batch_size,
                mixture_ratio=self.config.data.curriculum_mixture_ratio,
                replacement=True,  # Always use replacement for weighted sampling
                generator=torch.Generator().manual_seed(self.config.data.seed),
            )
            sampler = self.curriculum_sampler
        elif self.config.data.sampling_strategy == "shuffle":
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.seed)
            sampler = RandomSampler(
                data_source=self.train_dataset, generator=train_dataloader_generator
            )
        else:  # sequential
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.rollout_batch_size,
            sampler=sampler,
            num_workers=32,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True,
        )

        self.val_dataset = RLHFDataset(
            data_path=self.config.data.val_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            answer_key=self.config.data.answer_key,
            image_key=self.config.data.image_key,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation="right",
            system_prompt=self.config.data.system_prompt,
            min_pixels=self.config.data.min_pixels,
            max_pixels=self.config.data.max_pixels,
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=(
                len(self.val_dataset)
                if self.config.data.val_batch_size == -1
                else self.config.data.val_batch_size
            ),
            shuffle=False,
            num_workers=32,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
        )

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1
        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        print(f"Size of val dataloader: {len(self.val_dataloader)}")

        if self.config.trainer.max_steps is not None:
            training_steps = self.config.trainer.max_steps
        else:
            training_steps = (
                len(self.train_dataloader) * self.config.trainer.total_episodes
            )

        self.training_steps = training_steps
        self.config.worker.actor.optim.training_steps = training_steps
        self.config.worker.critic.optim.training_steps = training_steps
        print(f"Total training steps: {self.training_steps}")

    def _calculate_curriculum_metric(self, batch: DataProto) -> Dict:
        """Calculate the curriculum metric based on the configured strategy.
        
        Returns:
            A dictionary containing futures for the remote metric calculations
            and metadata needed to combine them later.
        """
        batch_size = len(batch.batch)

        # Generate responses for the batch - this has to be done sequentially
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

        gen_batch.meta_info["n"] = self.config.data.curriculum_rollout_n
        gen_batch.meta_info["disable_sleep"] = True
        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
        gen_batch.meta_info.pop("disable_sleep")
        gen_batch.meta_info.pop("n")

        batch = batch.repeat(
            repeat_times=self.config.data.curriculum_rollout_n, interleave=True
        )
        batch = batch.union(gen_batch_output)

        # Launch parallel metric computation based on configured metrics
        metric_futures = []
        
        for metric_name in self.config.data.curriculum_metrics:
            if metric_name == "learnability":
                metric_futures.append(
                    calculate_learnability_metric.remote(
                        self.reward_fn, 
                        batch, 
                        batch_size, 
                        self.config.data.curriculum_rollout_n
                    )
                )
            elif metric_name == "distinct_3":
                metric_futures.append(
                    calculate_distinct_n_metric.remote(
                        gen_batch_output.batch["responses"],
                        batch_size,
                        self.config.data.curriculum_rollout_n,
                        n=3
                    )
                )
            elif metric_name == "self_bleu_123":
                metric_futures.append(
                    calculate_self_bleu_metric.remote(
                        gen_batch_output.batch["responses"],
                        batch_size,
                        self.config.data.curriculum_rollout_n
                    )
                )
            elif metric_name == "edit_distance":
                metric_futures.append(
                    calculate_edit_distance_metric.remote(
                        gen_batch_output.batch["responses"],
                        batch_size,
                        self.config.data.curriculum_rollout_n
                    )
                )
            else:
                raise ValueError(
                    f"Unknown curriculum metric: {metric_name}"
                )
        
        # Return the futures along with metadata needed to combine them later
        return {
            "futures": metric_futures,
            "weights": self.config.data.curriculum_metric_weights,
            "metrics": self.config.data.curriculum_metrics
        }
    
    def _save_curriculum_weights(self, weights: torch.Tensor, step: int = None) -> None:
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
        print(f"Saved curriculum weights to {weights_path}")

    def _compute_curriculum_weights(self) -> torch.Tensor:
        """Compute curriculum weights for each sample in the dataset by sequentially generating sequences
        but calculating metrics in parallel, without waiting for each batch's metrics to complete."""
        curriculum_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.curriculum_rollout_batch_size,
            sampler=PaddedSequentialSampler(
                self.train_dataset,
                batch_size=self.config.data.curriculum_rollout_batch_size,
            ),
            shuffle=False,
            num_workers=32,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False,
        )

        # Initialize weights tensor
        curriculum_weights = torch.zeros(
            len(curriculum_dataloader) * self.config.data.curriculum_rollout_batch_size,
            dtype=torch.float32,
        )
        
        # Process batches sequentially but don't wait for metric calculations
        pending_results = []  # Store (batch_idx, futures_dict) tuples
        
        # Generate sequences for all batches first, queueing metric calculations
        print("Generating sequences and launching metric calculations...")
        for batch_idx, batch_dict in enumerate(curriculum_dataloader):
            batch = DataProto.from_single_dict(batch_dict)
            
            # Calculate metrics, get back futures dict
            result_dict = self._calculate_curriculum_metric(batch)
            
            # Store batch index and futures for later collection
            pending_results.append((batch_idx, result_dict))
            
            # Log progress
            print(f"Processed {batch_idx + 1}/{len(curriculum_dataloader)} batches for sequence generation")
                
        # Now process all the pending futures
        print(f"Processing {len(pending_results)} pending metric calculations...")
        for batch_idx, result_dict in pending_results:
            futures = result_dict["futures"]
            weights = result_dict["weights"]
            metrics = result_dict["metrics"]
            
            # Collect results from futures
            metric_results = [ray.get(future) for future in futures]
            
            # Combine the metric results for this batch
            combined_metric = combine_metric_results(metric_results, weights, metrics)
            
            # Store the combined metric
            start_idx = batch_idx * self.config.data.curriculum_rollout_batch_size
            end_idx = start_idx + self.config.data.curriculum_rollout_batch_size
            curriculum_weights[start_idx:end_idx] = combined_metric.detach()
            
            # Log progress
            print(f"Processed metrics for {batch_idx + 1}/{len(pending_results)} batches")

        # Trim any extra padding indices
        curriculum_weights = curriculum_weights[: len(self.train_dataset)]

        del curriculum_dataloader

        # Normalize weights using min-max scaling
        min_weight = curriculum_weights.min()
        max_weight = curriculum_weights.max()
        curriculum_weights = (curriculum_weights - min_weight) / (
            max_weight - min_weight + 1e-8  # Add small epsilon to avoid division by zero
        )

        return curriculum_weights
    
    def _init_curriculum_weights(self) -> None:
        """Initialize curriculum weights."""
        print("Initializing curriculum weights using parallel computation...")
        self.curriculum_weights = self._compute_curriculum_weights()

        # Store weights in dataset for access during training
        self.train_dataset.curriculum_weights = self.curriculum_weights

        # Save initial weights
        self._save_curriculum_weights(self.curriculum_weights)
        print("Curriculum weights initialized.")

    def _update_curriculum_weights(self) -> None:
        """Update curriculum learning weights based on current model performance."""
        print(f"Updating curriculum weights at step {self.global_step}...")
        # Create a temporary dataloader for weight estimation
        new_weights = self._compute_curriculum_weights()

        # Update weights with configured momentum
        momentum = self.config.data.curriculum_momentum
        self.curriculum_weights = (
            momentum * self.curriculum_weights + (1 - momentum) * new_weights
        )

        # Update sampler weights
        self.curriculum_sampler.update_weights(self.curriculum_weights)

        # Store updated weights in dataset
        self.train_dataset.curriculum_weights = self.curriculum_weights

        # Save updated weights
        self._save_curriculum_weights(self.curriculum_weights, step=self.global_step)

        # Create a new dataloader with the updated sampler
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.rollout_batch_size,
            sampler=self.curriculum_sampler,  # Using the updated sampler
            num_workers=32,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True,
        )

        # Set flag to signal that the dataloader iterator needs to be refreshed
        self.refresh_dataloader_iterator = True
        print(
            f"Curriculum weights updated at step {self.global_step}, dataloader will refresh for next batch"
        )

    def _maybe_log_val_generations(
        self, inputs: List[str], outputs: List[str], scores: List[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> Dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_scores = [], [], []
        reward_metrics_lst = defaultdict(list)
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in input_ids
            ]
            sample_inputs.extend(input_texts)

            if "multi_modal_inputs" in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=[
                        "raw_prompt_ids",
                        "multi_modal_data",
                        "multi_modal_inputs",
                    ],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, self.actor_rollout_wg.world_size
            )
            test_output_gen_batch = self.actor_rollout_wg.generate_sequences(
                test_gen_batch
            )
            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch, pad_size=pad_size
            )
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in output_ids
            ]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = self.val_reward_fn(test_batch)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

        self._maybe_log_val_generations(
            inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores
        )
        reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {
            f"val/{key}_reward": value
            for key, value in reduce_metrics(reward_metrics_lst).items()
        }
        return {"val/reward_score": reward_score, **val_reward_metrics}

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.ActorRollout
            )
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.worker,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool][
                "actor_rollout"
            ] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic],
                config=self.config.worker,
                role="critic",
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.worker,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.RewardModel
            )
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel],
                config=self.config.worker,
                role="reward",
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: Dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(
            self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}"
        )
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_wg.save_checkpoint(actor_path)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        last_global_step_path = os.path.join(
            self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER
        )
        with open(last_global_step_path, "w") as f:
            f.write(str(self.global_step))

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            pass
        elif osp.exists(self.config.trainer.save_checkpoint_path):
            ckpt_list = [
                _
                for _ in os.listdir(self.config.trainer.save_checkpoint_path)
                if _.startswith("global_step_")
            ]
            if len(ckpt_list) == 0:
                print(
                    f"No checkpoint found at {self.config.trainer.save_checkpoint_path}, will start from scratch."
                )
                return
            ckpt_list.sort(key=lambda x: int(x.split("global_step_")[-1]))
            self.config.trainer.load_checkpoint_path = os.path.join(
                self.config.trainer.save_checkpoint_path, ckpt_list[-1]
            )
        else:
            print(
                f"No checkpoint found at {self.config.trainer.save_checkpoint_path}, will start from scratch."
            )
            return

        if (
            "global_step_"
            not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(
                os.path.sep
            )[-1]
        ):
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        self.global_step = int(
            self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(
                "global_step_"
            )[-1]
        )
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(
                self.config.trainer.load_checkpoint_path, "critic"
            )
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(
            self.config.trainer.load_checkpoint_path, "dataloader.pt"
        )
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(
                f"No dataloader state found at {dataloader_path}, will start from scratch."
            )

    def _balance_batch(
        self,
        batch: DataProto,
        metrics: Dict[str, Any],
        logging_prefix: str = "global_seqlen",
    ) -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = (
            batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()
        )  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor(
            [j for partition in global_partition_lst for j in partition]
        )
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst,
            partitions=global_partition_lst,
            prefix=logging_prefix,
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        # breakpoint()
        self.logger = Tracker(
            loggers=self.config.trainer.logger, config=self.config.to_dict()
        )
        self.global_step = 0
        val_metrics: Optional[Dict[str, Any]] = None

        # Flag to track when to refresh the dataloader iterator
        self.refresh_dataloader_iterator = False

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        for epoch in tqdm(
            range(self.config.trainer.total_episodes), desc="Episode", position=0
        ):
            # Reset cached indices to ensure new indices are generated with the correct random position
            if self.config.data.sampling_strategy == "curriculum":
                self.curriculum_sampler.cached_mixed_indices = None
                print(
                    f"Reset cached indices for epoch {epoch}, random position: {self.curriculum_sampler.get_training_random_position()}"
                )

            # Create a new iterator for each epoch
            dataloader_iterator = iter(self.train_dataloader)

            # Loop until we've processed all batches or need to refresh the iterator
            while True:
                try:
                    # Get the next batch from the current iterator
                    if self.refresh_dataloader_iterator:
                        # If we need to refresh, create a new iterator with updated weights
                        dataloader_iterator = iter(self.train_dataloader)
                        self.refresh_dataloader_iterator = False
                        print("Dataloader iterator refreshed with updated weights")

                    batch_dict = next(dataloader_iterator)

                    self.global_step += 1
                    if self.global_step > self.training_steps:
                        break

                    metrics, timing_raw = {}, {}
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    # breakpoint()
                    # pop those keys for generation
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

                    with _timer("step", timing_raw):
                        # generate a batch
                        with _timer("gen", timing_raw):  # wg: worker group
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(
                                gen_batch
                            )

                        if self.config.algorithm.adv_estimator == "remax":
                            with _timer("gen_max", timing_raw):
                                gen_baseline_batch = deepcopy(gen_batch)
                                gen_baseline_batch.meta_info["temperature"] = 0.0
                                gen_baseline_output = (
                                    self.actor_rollout_wg.generate_sequences(
                                        gen_baseline_batch
                                    )
                                )

                                batch = batch.union(gen_baseline_output)
                                reward_baseline_tensor, _ = self.reward_fn(batch)
                                reward_baseline_tensor = reward_baseline_tensor.sum(
                                    dim=-1
                                )

                                batch.pop(
                                    batch_keys=list(gen_baseline_output.batch.keys())
                                )
                                batch.batch["reward_baselines"] = reward_baseline_tensor
                                del gen_baseline_batch, gen_baseline_output

                        batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                            dtype=object,
                        )
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(
                            repeat_times=self.config.worker.rollout.n, interleave=True
                        )
                        batch = batch.union(gen_batch_output)

                        # compute reward
                        with _timer("reward", timing_raw):
                            if self.use_reward_model:
                                raise NotImplementedError(
                                    "Reward model is not supported yet."
                                )

                            # we combine with rule-based rm
                            reward_tensor, reward_metrics = self.reward_fn(batch)
                            batch.batch["token_level_scores"] = reward_tensor
                            reward_metrics = {
                                f"reward/{key}": value
                                for key, value in reduce_metrics(reward_metrics).items()
                            }
                            metrics.update(reward_metrics)

                        # balance the number of valid tokens on each dp rank.
                        # Note that this breaks the order of data inside the batch.
                        # Please take care when you implement group based adv computation such as GRPO and rloo
                        self._balance_batch(batch, metrics=metrics)

                        # compute global_valid tokens
                        batch.meta_info["global_token_num"] = torch.sum(
                            batch.batch["attention_mask"], dim=-1
                        ).tolist()

                        # recompute old_log_probs
                        with _timer("old", timing_raw):
                            old_log_probs = self.actor_rollout_wg.compute_log_probs(
                                batch
                            )
                            batch = batch.union(old_log_probs)

                        # compute ref_log_probs
                        if self.use_reference_policy:
                            with _timer("ref", timing_raw):
                                ref_log_probs = (
                                    self.ref_policy_wg.compute_ref_log_probs(batch)
                                )
                                batch = batch.union(ref_log_probs)

                        # compute values
                        if self.use_critic:
                            with _timer("values", timing_raw):
                                values = self.critic_wg.compute_values(batch)
                                batch = batch.union(values)

                        with _timer("adv", timing_raw):
                            # apply kl penalty if available
                            if (
                                not self.config.algorithm.use_kl_loss
                                and self.use_reference_policy
                            ):  # apply kl penalty to reward
                                batch, kl_metrics = apply_kl_penalty(
                                    batch,
                                    kl_ctrl=self.kl_ctrl,
                                    kl_penalty=self.config.algorithm.kl_penalty,
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch[
                                    "token_level_scores"
                                ]

                            # compute advantages, executed on the driver process
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                            )

                        # update critic
                        if self.use_critic:
                            with _timer("update_critic", timing_raw):
                                critic_output = self.critic_wg.update_critic(batch)

                            critic_metrics = reduce_metrics(
                                critic_output.non_tensor_batch
                            )
                            metrics.update(critic_metrics)

                        # update actor
                        if self.config.trainer.critic_warmup <= self.global_step:
                            with _timer("update_actor", timing_raw):
                                actor_output = self.actor_rollout_wg.update_actor(batch)

                            actor_metrics = reduce_metrics(
                                actor_output.non_tensor_batch
                            )
                            metrics.update(actor_metrics)

                        # validate
                        if (
                            self.val_reward_fn is not None
                            and self.config.trainer.val_freq > 0
                            and self.global_step % self.config.trainer.val_freq == 0
                        ):
                            with _timer("validation", timing_raw):
                                val_metrics = self._validate()

                            metrics.update(val_metrics)

                        if (
                            self.config.trainer.save_freq > 0
                            and self.global_step % self.config.trainer.save_freq == 0
                        ):
                            with _timer("save_checkpoint", timing_raw):
                                self._save_checkpoint()

                        # Update curriculum weights if configured for step-level updates
                        if (
                            self.config.data.sampling_strategy == "curriculum"
                            and self.config.data.curriculum_update_freq > 0
                            and self.global_step
                            % self.config.data.curriculum_update_freq
                            == 0
                        ):
                            with _timer("update_curriculum", timing_raw):
                                self._update_curriculum_weights()

                                # Log curriculum learning metrics
                                curriculum_metrics = {
                                    "curriculum/mean_weight": self.curriculum_weights.mean().item(),
                                    "curriculum/std_weight": self.curriculum_weights.std().item(),
                                    "curriculum/min_weight": self.curriculum_weights.min().item(),
                                    "curriculum/max_weight": self.curriculum_weights.max().item(),
                                    "curriculum/consumed_batches": self.curriculum_sampler.consumed_batches,
                                    "curriculum/random_position": self.curriculum_sampler.get_training_random_position(),
                                }
                                metrics.update(curriculum_metrics)

                    # collect metrics
                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    metrics.update(
                        compute_data_metrics(batch=batch, use_critic=self.use_critic)
                    )
                    metrics.update(
                        compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                    )
                    metrics.update(
                        compute_throughout_metrics(
                            batch=batch, timing_raw=timing_raw, n_gpus=n_gpus
                        )
                    )

                    self.logger.log(data=metrics, step=self.global_step)

                    # Update the random indices position to properly track the samples seen
                    if self.config.data.sampling_strategy == "curriculum":
                        self.curriculum_sampler.update_position_after_batch()

                except StopIteration:
                    # We've reached the end of the current iterator
                    break

            # Exit the epoch loop if we've exceeded training steps
            if self.global_step > self.training_steps:
                break

            # Update curriculum weights at the end of each epoch if configured for epoch-level updates
            if (
                self.config.data.sampling_strategy == "curriculum"
                and self.config.data.curriculum_update_freq == 0
            ):
                with _timer("update_curriculum", timing_raw):
                    self._update_curriculum_weights()

                    # Log curriculum learning metrics
                    curriculum_metrics = {
                        "curriculum/mean_weight": self.curriculum_weights.mean().item(),
                        "curriculum/std_weight": self.curriculum_weights.std().item(),
                        "curriculum/min_weight": self.curriculum_weights.min().item(),
                        "curriculum/max_weight": self.curriculum_weights.max().item(),
                        "curriculum/consumed_batches": self.curriculum_sampler.consumed_batches,
                        "curriculum/random_position": self.curriculum_sampler.get_training_random_position(),
                    }
                    self.logger.log(data=curriculum_metrics, step=self.global_step)

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics: {convert_dict_to_str(val_metrics)}")

        if (
            self.config.trainer.save_freq <= 0
            or self.global_step % self.config.trainer.save_freq != 0
        ):
            self._save_checkpoint()


def calculate_distinct_n(tokenized_sequences, n):
    """Calculate distinct-n metric using model's tokenized sequences."""
    all_ngrams = []
    for seq in tokenized_sequences:
        all_ngrams.extend(list(zip(*[seq[i:] for i in range(n)])))

    if not all_ngrams:
        return 0.0

    unique_ngrams = set(all_ngrams)
    return len(unique_ngrams) / len(all_ngrams)


def calculate_self_bleu_n(tokenized_sequences, n):
    """Calculate self-BLEU score for specific n-gram using model's tokenized sequences."""
    if len(tokenized_sequences) < 2:
        return 0.0

    weights = tuple([1.0] if n == 1 else [0.0] * (n - 1) + [1.0])  # Only use n-gram
    scores = []

    for i, hyp in enumerate(tokenized_sequences):
        refs = tokenized_sequences[:i] + tokenized_sequences[i + 1 :]
        score = sentence_bleu(
            refs, hyp, weights=weights, smoothing_function=SmoothingFunction().method1
        )
        scores.append(score)

    return np.mean(scores)


def calculate_self_bleu_123(tokenized_sequences):
    """Calculate self-BLEU-123 score with uniform weights using model's tokenized sequences."""
    if len(tokenized_sequences) < 2:
        return 0.0

    weights = (1 / 3, 1 / 3, 1 / 3)  # Uniform weights for 1,2,3-grams
    scores = []

    for i, hyp in enumerate(tokenized_sequences):
        refs = tokenized_sequences[:i] + tokenized_sequences[i + 1 :]
        score = sentence_bleu(
            refs, hyp, weights=weights, smoothing_function=SmoothingFunction().method1
        )
        scores.append(score)

    return np.mean(scores)


def calculate_pairwise_edit_distance(tokenized_sequences):
    """Calculate average pairwise edit distance between all tokenized sequences."""
    if len(tokenized_sequences) < 2:
        return 0.0

    distances = []
    for i in range(len(tokenized_sequences)):
        for j in range(i + 1, len(tokenized_sequences)):
            dist = distance(tokenized_sequences[i], tokenized_sequences[j])
            distances.append(dist)

    return np.mean(distances) if distances else 0.0


# Helper function for multiprocessing BLEU score calculation
def calculate_batch_self_bleu_123_helper(batch_idx, tokens, n_per_item):
    """Calculate self-BLEU-123 score for a batch segment."""
    start_idx = batch_idx * n_per_item
    end_idx = (batch_idx + 1) * n_per_item
    item_tokens = tokens[start_idx:end_idx]
    return calculate_self_bleu_123(item_tokens)


def calculate_pairwise_edit_distance_helper(batch_idx, tokens, n_per_item):
    """Calculate average pairwise edit distance between all tokenized sequences."""
    start_idx = batch_idx * n_per_item
    end_idx = (batch_idx + 1) * n_per_item
    item_tokens = tokens[start_idx:end_idx]
    return calculate_pairwise_edit_distance(item_tokens)
