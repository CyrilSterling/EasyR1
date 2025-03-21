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


import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from ...utils.reward_score import math_compute_score, r1v_compute_score, openr1_compute_score, openr1_compute_score_batch


class CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, num_examine: int, compute_score: str, validation: bool, response_length: int = None, batch_processing: bool = False):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.validation = validation
        self.response_length = response_length
        self.batch_processing = batch_processing
        if compute_score == "math":
            self.compute_score = math_compute_score
        elif compute_score == "r1v":
            self.compute_score = r1v_compute_score
        elif compute_score == "openr1":
            if self.batch_processing:
                self.compute_score = openr1_compute_score_batch
            else:
                self.compute_score = openr1_compute_score
        else:
            raise NotImplementedError()

    def batch_process(self, data: DataProto, reward_tensor: torch.Tensor, already_print: int) -> torch.Tensor:
        breakpoint()
        prompt_strs = []
        response_strs = []
        ground_truths = []
        valid_response_lengths = []
        for i in range(len(data)):
            data_item = data[i]
            
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            prompt_strs.append(prompt_str)
            response_strs.append(response_str)
            ground_truths.append(data_item.non_tensor_batch["ground_truth"])
            valid_response_lengths.append(valid_response_length)
        
        scores = self.compute_score(response_strs, ground_truths, prompt_strs, self.validation, self.response_length)
        for i in range(len(data)):
            reward_tensor[i, valid_response_lengths[i] - 1] = scores[i]
            if already_print < self.num_examine:
                already_print += 1
                print("[prompt]", prompt_strs[i])
                print("[response]", response_strs[i])
                print("[ground_truth]", ground_truths[i])
                print("[score]", scores[i])
        return reward_tensor
                
            
    
    def __call__(self, data: DataProto) -> torch.Tensor:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        already_print = 0

        if self.batch_processing:
            return self.batch_process(data, reward_tensor, already_print)
        else:
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                ground_truth = data_item.non_tensor_batch["ground_truth"]
                
                ## TODO: batched compute score for LMM as judge efficiency (try gpt first then vllm deployment)
                score = self.compute_score(response_str, ground_truth, self.validation, self.response_length)
                reward_tensor[i, valid_response_length - 1] = score

                if already_print < self.num_examine:
                    already_print += 1
                    print("[prompt]", prompt_str)
                    print("[response]", response_str)
                    print("[ground_truth]", ground_truth)
                    print("[score]", score)

            return reward_tensor
