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

import os
import os.path as osp
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Callable, Dict, List, Optional, Type, Tuple

import numpy as np
import ray
import ray.exceptions
import torch
from codetiming import Timer
from PIL import Image
from ray.experimental.tqdm_ray import tqdm
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..curriculum import CurriculumManager
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
from .config import (
    ActorRolloutConfig,
    AdvantageEstimator,
    PPOConfig,
    RefPolicyConfig,
    RewardModelConfig,
)
from ..workers.reward import CustomRewardManager


class WorkerType(IntEnum):
    """Worker type used for clarity and worker initialization"""

    ACTOR = 0
    CRITIC = 1
    REF_POLICY = 2
    REWARD_MODEL = 3
    ACTOR_ROLLOUT = 4  # the hybrid engine
    DUMMY = 100  # never used in real experiments


class Role(Enum):
    """The role of workers"""

    Policy = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    Dummy = auto()  # for testing


@contextmanager
def _timer(name, timing_raw):
    timer = Timer(name, logger=None)
    timer.start()
    yield
    timer.stop()
    timing_raw[name] = timer.last


class ResourcePoolManager:
    """Manager resource pools for different roles."""

    def __init__(
        self,
        role_worker_mapping: Dict[Role, Type[Worker]],
        use_hybrid_engine: bool = True,
        use_reference_policy: bool = True,
        use_critic: bool = True,
        use_reward_model: bool = False,
        actor_config: ActorRolloutConfig = None,
        critic_config: ActorRolloutConfig = None,
        ref_policy_config: RefPolicyConfig = None,
        reward_model_config: RewardModelConfig = None,
    ):
        """Init the ResourcePoolManager.

        Args:
            role_worker_mapping: mapping from role to worker class.
            actor_config: actor configuration.
            critic_config: critic configuration.
            ref_policy_config: reference policy configuration.
            reward_model_config: reward model configuration.
        """
        self.role_worker_mapping = role_worker_mapping
        self.use_hybrid_engine = use_hybrid_engine
        self.use_reference_policy = use_reference_policy
        self.use_critic = use_critic
        self.use_reward_model = use_reward_model
        self.actor_config = actor_config
        self.critic_config = critic_config
        self.ref_policy_config = ref_policy_config
        self.reward_model_config = reward_model_config

    def create_resource_pool(self):
        """Create resource pools for different roles.
        We have the following resource pools:
        - ActorRollout (hybrid engine): Used for actor rollout with generation and training.
        - Policy (non-hybrid engine): Used for actor training only
        - Critic: Used for critic training
        - RefPolicy: Used for computing reference log probs
        - RewardModel: Used for computing reward

        Not all resource pools are required.
        Hybrid engine: ActorRollout, Critic, RefPolicy (optional), RewardModel (optional)
        Non-hybrid engine: Policy, Critic, RefPolicy (optional), RewardModel (optional)
        """
        # Rank 0 must be allocated to rollout in the model name mapping
        resource_pool_dict = {}

        if self.use_hybrid_engine:
            if Role.ActorRollout not in self.role_worker_mapping:
                raise ValueError("Hybrid engine requires actor rollout worker.")
            resource_pool_dict[Role.ActorRollout] = RayResourcePool(
                process_cls=RayClassWithInitArgs(
                    self.role_worker_mapping[Role.ActorRollout],
                    **self.actor_config.to_dict(),
                ),
                name=Role.ActorRollout.name,
                num_workers=self.actor_config.parallel_num,
                use_gpu=self.actor_config.use_gpu,
            )
        else:
            if Role.Policy not in self.role_worker_mapping:
                raise ValueError("Non-hybrid engine requires policy worker.")
            resource_pool_dict[Role.Policy] = RayResourcePool(
                process_cls=RayClassWithInitArgs(
                    self.role_worker_mapping[Role.Policy],
                    **self.actor_config.to_dict(),
                ),
                name=Role.Policy.name,
                num_workers=self.actor_config.parallel_num,
                use_gpu=self.actor_config.use_gpu,
            )

        if self.use_critic:
            if Role.Critic not in self.role_worker_mapping:
                raise ValueError("Critic worker is required if use_critic is True.")
            resource_pool_dict[Role.Critic] = RayResourcePool(
                process_cls=RayClassWithInitArgs(
                    self.role_worker_mapping[Role.Critic], **self.critic_config.to_dict()
                ),
                name=Role.Critic.name,
                num_workers=self.critic_config.parallel_num,
                use_gpu=self.critic_config.use_gpu,
            )

        if self.use_reference_policy:
            if Role.RefPolicy not in self.role_worker_mapping:
                raise ValueError(
                    "Reference policy worker is required if use_reference_policy is True."
                )
            resource_pool_dict[Role.RefPolicy] = RayResourcePool(
                process_cls=RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RefPolicy],
                    **self.ref_policy_config.to_dict(),
                ),
                name=Role.RefPolicy.name,
                num_workers=self.ref_policy_config.parallel_num,
                use_gpu=self.ref_policy_config.use_gpu,
            )

        if self.use_reward_model:
            if Role.RewardModel not in self.role_worker_mapping:
                raise ValueError(
                    "Reward model worker is required if use_reward_model is True."
                )
            resource_pool_dict[Role.RewardModel] = RayResourcePool(
                process_cls=RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RewardModel],
                    **self.reward_model_config.to_dict(),
                ),
                name=Role.RewardModel.name,
                num_workers=self.reward_model_config.parallel_num,
                use_gpu=self.reward_model_config.use_gpu,
            )

        self.resource_pool_dict = resource_pool_dict

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get resource pool by role."""
        return self.resource_pool_dict[role]

    def get_all_resource_pools(self) -> List[RayResourcePool]:
        """Get all resource pools."""
        return list(self.resource_pool_dict.values())


def _convert_str_to_args(config, mapping: Dict[str, Type]):
    """Internal function to convert string to class type"""
    for key, value in mapping.items():
        keys = key.split(".")
        current_config = config
        for k in keys[:-1]:
            current_config = getattr(current_config, k)
        setattr(current_config, keys[-1], value)


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

        self.role_worker_mapping = role_worker_mapping
        self.hybrid_engine = config.worker.hybrid_engine
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

        self.checkpoint_path = self.config.load_checkpoint_path = (
            self._get_checkpoint_path()
        )

        # Initialize workers first
        self.init_workers()

        self.global_step = 0
        # Load checkpoint for workers and dataloader
        self._load_worker_state()

        # Then create dataloader which depends on workers
        self._create_dataloader()

        # Initialize curriculum manager if needed
        self.curriculum_manager = None
        if self.config.data.sampling_strategy == "curriculum":
            self.curriculum_manager = CurriculumManager(
                config=self.config,
                train_dataset=self.train_dataset,
                actor_rollout_wg=self.actor_rollout_wg,
                reward_fn=self.reward_fn,
                checkpoint_path=self.checkpoint_path,
            )

        self.logger = Tracker(
            loggers=self.config.trainer.logger, config=self.config.to_dict()
        )

    def _create_dataloader(self) -> None:
        # --- Create Train Dataset ---
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

        # --- Create Sampler ---
        if self.config.data.sampling_strategy == "curriculum":
            if self.curriculum_manager is None:
                self.curriculum_manager = CurriculumManager(
                    config=self.config,
                    train_dataset=self.train_dataset,
                    actor_rollout_wg=self.actor_rollout_wg,
                    reward_fn=self.reward_fn,
                    checkpoint_path=self.checkpoint_path,
                )
            sampler = self.curriculum_manager.get_sampler()

        elif self.config.data.sampling_strategy == "shuffle":
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.seed)
            sampler = RandomSampler(
                data_source=self.train_dataset, generator=train_dataloader_generator
            )
        elif self.config.data.sampling_strategy == "sequential":
            sampler = SequentialSampler(data_source=self.train_dataset)
        else:
            raise NotImplementedError(
                f"Sampling strategy {self.config.data.sampling_strategy} is not implemented."
            )

        # --- Create Train Dataloader ---
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.rollout_batch_size,
            shuffle=False,  # shuffle is handled by sampler
            sampler=sampler,
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True,
        )

        # --- Create Val Dataset and Dataloader ---
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
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
        )

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        if self.config.trainer.max_steps is not None:
            training_steps = self.config.trainer.max_steps
        else:
            training_steps = (
                len(self.train_dataloader) * self.config.trainer.total_episodes
            )

        self.training_steps = training_steps
        self.config.worker.actor.optim.training_steps = training_steps
        self.config.worker.critic.optim.training_steps = training_steps

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

        # Take first N samples for logging
        n_samples = min(self.config.trainer.val_generations_to_log, len(samples))
        log_samples = samples[:n_samples]

        # Prepare table data
        table_data = []
        for inp, out, score in log_samples:
            table_data.append(
                {"Input": inp[:200], "Output": out[:500], "Score": f"{score:.3f}"}
            )

        self.logger.log(data={"val/generations": table_data}, step=self.global_step)

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        reward_metrics_lst = defaultdict(list)
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_batch_dict in tqdm(
            self.val_dataloader, desc="Validation", position=1
        ):
            test_batch: DataProto = DataProto.from_single_dict(test_batch_dict)
            # Store prompt texts for logging
            input_ids = test_batch.batch["input_ids"]
            input_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids
            ]
            sample_inputs.extend(input_texts)

            # pop those keys for generation
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

            # Generate a batch
            pad_size = 0
            test_output_gen_batch = self.actor_rollout_wg.generate_sequences(
                test_gen_batch, add_padding_to_divisor=True
            )
            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch, pad_size=pad_size
            )

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
                config=self.config.worker.actor_rollout.to_dict(),
            )
            self.resource_pool_to_cls[resource_pool][Role.ActorRollout] = (
                actor_rollout_cls
            )

            self.actor_rollout_wg = self.ray_worker_group_cls(
                resource_pool=resource_pool, ray_cls_with_init=actor_rollout_cls
            )
        else:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Policy)
            policy_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Policy],
                config=self.config.worker.actor.to_dict(),
            )
            self.resource_pool_to_cls[resource_pool][Role.Policy] = policy_cls

            self.policy_wg = self.ray_worker_group_cls(
                resource_pool=resource_pool, ray_cls_with_init=policy_cls
            )

            self.actor_rollout_wg = self.policy_wg

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic],
                config=self.config.worker.critic.to_dict(),
            )
            self.resource_pool_to_cls[resource_pool][Role.Critic] = critic_cls

            self.critic_wg = self.ray_worker_group_cls(
                resource_pool=resource_pool, ray_cls_with_init=critic_cls
            )

        # create reference policy
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RefPolicy],
                config=self.config.worker.ref.to_dict(),
            )
            self.resource_pool_to_cls[resource_pool][Role.RefPolicy] = ref_policy_cls

            self.ref_policy_wg = self.ray_worker_group_cls(
                resource_pool=resource_pool, ray_cls_with_init=ref_policy_cls
            )

        # create reward model
        if self.use_reward_model:
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.RewardModel
            )
            reward_model_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel],
                config=self.config.worker.reward_model.to_dict(),
            )
            self.resource_pool_to_cls[resource_pool][Role.RewardModel] = (
                reward_model_cls
            )

            self.reward_model_wg = self.ray_worker_group_cls(
                resource_pool=resource_pool, ray_cls_with_init=reward_model_cls
            )

    def _save_checkpoint(self):
        """Save checkpoint to a specific location and potentially remove obsolete checkpoints"""
        folder_path = os.path.join(
            self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}"
        )

        # save workers
        if self.hybrid_engine:
            self.actor_rollout_wg.save_checkpoint(
                tag=f"global_step_{self.global_step}",
                path=self.config.trainer.save_checkpoint_path,
            )
        else:
            self.policy_wg.save_checkpoint(
                tag=f"global_step_{self.global_step}",
                path=self.config.trainer.save_checkpoint_path,
            )

        if self.use_critic:
            self.critic_wg.save_checkpoint(
                tag=f"global_step_{self.global_step}",
                path=self.config.trainer.save_checkpoint_path,
            )

        if self.use_reference_policy:
            self.ref_policy_wg.save_checkpoint(
                tag=f"global_step_{self.global_step}",
                path=self.config.trainer.save_checkpoint_path,
            )

        if self.use_reward_model:
            self.reward_model_wg.save_checkpoint(
                tag=f"global_step_{self.global_step}",
                path=self.config.trainer.save_checkpoint_path,
            )

        # save train_dataloader
        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        torch.save(self.train_dataloader.state_dict(), dataloader_path)

        # Save curriculum learning state if using curriculum strategy
        if self.curriculum_manager is not None:
            self.curriculum_manager.save_state(folder_path)

        last_global_step_path = os.path.join(
            self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER
        )
        with open(last_global_step_path, "w") as f:
            f.write(str(self.global_step))

    def _get_checkpoint_path(self) -> str:
        if self.config.trainer.load_checkpoint_path is not None:
            return None
        if not os.path.exists(self.config.trainer.save_checkpoint_path):
            return None

        ckpt_list = [
            _
            for _ in os.listdir(self.config.trainer.save_checkpoint_path)
            if _.startswith("global_step_")
        ]
        if len(ckpt_list) == 0:
            return None

        ckpt_list.sort(key=lambda x: int(x.split("global_step_")[-1]))
        return os.path.join(self.config.trainer.save_checkpoint_path, ckpt_list[-1])

    def _load_dataloader_state(self) -> None:
        if self.checkpoint_path is None:
            return

        dataloader_path = os.path.join(self.checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)

    def _load_worker_state(self) -> None:
        if self.checkpoint_path is None:
            return

        self.global_step = int(
            self.checkpoint_path.strip(os.path.sep).split("global_step_")[-1]
        )

        # load workers
        if self.hybrid_engine:
            self.actor_rollout_wg.load_checkpoint(
                tag=f"global_step_{self.global_step}",
                path=self.config.trainer.save_checkpoint_path,
            )
        else:
            self.policy_wg.load_checkpoint(
                tag=f"global_step_{self.global_step}",
                path=self.config.trainer.save_checkpoint_path,
            )

        if self.use_critic:
            self.critic_wg.load_checkpoint(
                tag=f"global_step_{self.global_step}",
                path=self.config.trainer.save_checkpoint_path,
            )

        if self.use_reference_policy:
            self.ref_policy_wg.load_checkpoint(
                tag=f"global_step_{self.global_step}",
                path=self.config.trainer.save_checkpoint_path,
            )

        if self.use_reward_model:
            self.reward_model_wg.load_checkpoint(
                tag=f"global_step_{self.global_step}",
                path=self.config.trainer.save_checkpoint_path,
            )

    def fit(self):
        """Main training loop with mixed PPO steps."""
        val_metrics: dict[str, Any] | None = None

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
            if self.curriculum_manager is not None:
                self.curriculum_manager.reset_cached_indices()

            # Create a new iterator for each epoch
            self.dataloader_iterator = iter(self.train_dataloader)

            # Loop until we've processed all batches or need to refresh the iterator
            while True:
                try:
                    batch_dict = next(self.dataloader_iterator)

                    self.global_step += 1
                    if self.global_step > self.training_steps:
                        break

                    metrics, timing_raw = {}, {}
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
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
                                f"reward/{k}": v for k, v in reward_metrics.items()
                            }

                        # Compute KL divergence
                        with _timer("ref_log_probs", timing_raw):
                            if self.use_reference_policy:
                                ref_log_probs, ref_output = (
                                    self.ref_policy_wg.compute_log_probs(batch)
                                )
                                # Update batch - actor will need to recalculate if HybridEngine
                                batch.batch["ref_log_probs"] = ref_log_probs

                        # compute values if use critic, otherwise heuristics are used
                        with _timer("values", timing_raw):
                            if self.use_critic:
                                values, value_output = self.critic_wg.compute_values(batch)
                                batch.batch["values"] = values

                        # Update actors
                        with _timer("update_actor", timing_raw):
                            kl_result = {"kl_ctl/value": self.kl_ctrl.value}
                            actor_output, batch_with_generate_data, non_tensor_dict = (
                                self.actor_rollout_wg.update_actor(
                                    batch, self.kl_ctrl, **kl_result
                                )
                            )
                            actor_metrics = reduce_micro_batch_metrics(
                                actor_output.metrics
                            )

                        # update KL controller
                        if self.use_reference_policy:
                            kl_mean = actor_metrics["rl/kl_mean"]
                            self.kl_ctrl.update(kl_mean)
                            kl_result["kl_ctl/value_updated"] = self.kl_ctrl.value

                        # update critic
                        with _timer("update_critic", timing_raw):
                            if self.use_critic:
                                critic_output = self.critic_wg.update_critic(batch)
                                critic_metrics = reduce_micro_batch_metrics(
                                    critic_output.metrics
                                )

                        # Update curriculum weights if configured for step-level updates
                        if (
                            self.curriculum_manager is not None
                            and self.curriculum_manager.should_update_weights(self.global_step)
                        ):
                            with _timer("update_curriculum", timing_raw):
                                self.curriculum_manager.update_weights(self.global_step)

                                # Log curriculum learning metrics
                                curriculum_metrics = self.curriculum_manager.get_metrics()
                                metrics.update(curriculum_metrics)

                        # logging
                        metrics.update(timing_raw)
                        metrics.update(actor_metrics)
                        metrics.update(critic_metrics) if self.use_critic else None
                        metrics.update(kl_result)
                        metrics.update(reward_metrics)

                        # save to storage
                        if (
                            self.config.trainer.save_freq > 0
                            and self.global_step % self.config.trainer.save_freq == 0
                        ):
                            self._save_checkpoint()
                            remove_obsolete_ckpt(
                                self.config.trainer.save_checkpoint_path,
                                self.config.trainer.max_save,
                            )

                        # remote validation
                        if (
                            self.val_reward_fn is not None
                            and self.config.trainer.val_freq > 0
                            and self.global_step % self.config.trainer.val_freq == 0
                        ):
                            val_metrics = self._validate()
                            metrics.update(val_metrics)

                        # Logging
                        self.logger.log(data=metrics, step=self.global_step)

                    if self.curriculum_manager is not None:
                        self.curriculum_manager.update_position_after_batch()

                except StopIteration:
                    # If we've consumed all batches for this epoch, break to start new epoch
                    break

            # Check if we've exceeded max steps
            if self.global_step > self.training_steps:
                break

            # Update curriculum weights at the end of each epoch if configured for epoch-level updates
            if (
                self.curriculum_manager is not None
                and self.curriculum_manager.should_update_weights_epoch()
            ):
                with _timer("update_curriculum", timing_raw):
                    self.curriculum_manager.update_weights(self.global_step)

                    # Log curriculum learning metrics
                    curriculum_metrics = self.curriculum_manager.get_metrics()
                    self.logger.log(data=curriculum_metrics, step=self.global_step)

        # perform validation after training if not already done
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

        if (
            self.config.trainer.save_freq <= 0
            or self.global_step % self.config.trainer.save_freq != 0
        ):
            self._save_checkpoint()


def reduce_metrics(all_metrics: Dict[str, List[torch.Tensor]]) -> Dict[str, Any]:
    """
    average over a list of 1-d tensors/lists
    """
    reduced_metrics = {}
    for key, value in all_metrics.items():
        value = torch.tensor(value, dtype=torch.float32)
        reduced_metrics[key] = value.mean().item()
    return reduced_metrics


def reduce_micro_batch_metrics(raw_metrics):
    metrics = defaultdict(list)
    for raw_metric in raw_metrics:
        for key, value in raw_metric.items():
            if isinstance(value, torch.Tensor):
                metrics[key].append(value.mean().item())
            else:
                metrics[key].append(value)
    metrics = {key: np.mean(value) for key, value in metrics.items()}
    return metrics