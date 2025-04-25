"""Reward functions for GRPO training."""

import asyncio
import json
import math
import re
from typing import Dict

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from .r1v import r1v_format_reward, r1v_accuracy_reward, r1v_accuracy_only_reward
from .gpt_as_judge import openai_llm, get_compare_messages

from mathruler.grader import extract_boxed_content

def format_reward_batch(completion_contents, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def get_cosine_scaled_reward_batch(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.0, # min_value_correct: float = 0.5
    max_value_correct: float = 0.0, # max_value_correct: float = 1.0
    max_len: int = 2048,
):
    def cosine_scaled_reward(completions, solution, acc_rewards, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = completions
        rewards = []

        for content, sol, acc_reward in zip(contents, solution, acc_rewards):
            is_correct = float(acc_reward) == 1.0
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward



def get_repetition_penalty_reward(ngram_size: int=40, max_penalty: float=-0.5):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = completions
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward

def accuracy_reward_batch_w_LMM_as_judge(predict_strs, ground_truths, prompt_strs, response_length):
    acc_rewards = []
    for predict_str, ground_truth in zip(predict_strs, ground_truths):
        acc_reward = r1v_accuracy_reward(predict_str, ground_truth, response_length)
        acc_rewards.append(acc_reward)
    # call gpt as judge for acc_reward if not 1.0
    llm_client = openai_llm()
    # find the index of acc_reward that is not 1.0
    idxs = [i for i, acc_reward in enumerate(acc_rewards) if float(acc_reward) != 1.0]
    message_list = []
    # process prompt_strs. the question is between \nuser\n and \nassistant\n
    question_strs = [prompt_str.split("\nuser\n")[1].split("\nassistant\n")[0].strip() for prompt_str in prompt_strs]
    # process answer_strs.
    answer_matches = [re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL) for predict_str in predict_strs]
    answer_strs_ = [answer_match.group(1).strip() if answer_match else predict_str.strip() for answer_match, predict_str in zip(answer_matches, predict_strs)]
    answer_strs = []
    for answer_str in answer_strs_:
        if "box" in answer_str:
            answer_str = extract_boxed_content(answer_str).strip()
        answer_strs.append(answer_str)
    for ind in idxs:
        message = get_compare_messages(question_strs[ind], answer_strs[ind], ground_truths[ind])
        message_list.append(message)
    gpt_outputs = llm_client.generate_outputs(message_list)
    gpt_corrects_num = 0
    for i, gpt_output in enumerate(gpt_outputs):
        # find the content between <judge> and </judge> tags
        if re.search(r"<judge>.*</judge>", gpt_output):
            judge_content = re.search(r"<judge>.*</judge>", gpt_output).group()
        else:
            judge_content = "<judge>1</judge>"
        if judge_content.strip() == "<judge>0</judge>": # judge as correct, replace the reward as 1.0
            acc_rewards[idxs[i]] = 1.0
            gpt_corrects_num += 1
        else:
            continue
    print(f"Corrected {gpt_corrects_num}/{len(idxs)} rewards using GPT as judge.")
    return acc_rewards

def accuracy_reward_batch_vllm(predict_strs, ground_truths, prompt_strs, response_length, **kwargs):
    """Calculate accuracy rewards using vLLM as the judge.
    
    Args:
        predict_strs: List of model predictions 
        ground_truths: List of ground truth answers
        prompt_strs: List of prompt strings
        response_length: Maximum response length
        **kwargs: Additional arguments
            provider: The LLM provider type ("vllm")
            base_urls: List of URLs for vLLM servers for parallel processing
            model_name: The name of the model to use
            api_key: API key for the vLLM server (can be dummy for vLLM)
    """
    acc_rewards = []
    for predict_str, ground_truth in zip(predict_strs, ground_truths):
        acc_reward = r1v_accuracy_only_reward(predict_str, ground_truth, response_length)
        acc_rewards.append(acc_reward)
    
    # Find the index of acc_reward that is not 1.0
    idxs = [i for i, acc_reward in enumerate(acc_rewards) if float(acc_reward) != 1.0]
    if not idxs:  # If all rewards are already 1.0, return early
        return acc_rewards
        
    # Process prompt_strs. The question is between \nuser\n and \nassistant\n
    question_strs = [prompt_str.split("\nuser\n")[1].split("\nassistant\n")[0].strip() for prompt_str in prompt_strs]
    
    # Process answer_strs
    answer_matches = [re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL) for predict_str in predict_strs]
    answer_strs_ = [answer_match.group(1).strip() if answer_match else predict_str.strip() for answer_match, predict_str in zip(answer_matches, predict_strs)]
    answer_strs = []
    for answer_str in answer_strs_:
        if "boxed" in answer_str:
            answer_str = extract_boxed_content(answer_str).strip()
        answer_strs.append(answer_str)
    
    # Get base_urls - either a list or convert single URL to a list
    base_urls = kwargs.get("base_urls", [kwargs.get("base_url", "http://localhost:8000/v1")])
    if not isinstance(base_urls, list):
        base_urls = [base_urls]
    
    model_name = kwargs.get("model_name", "NousResearch/Meta-Llama-3-8B-Instruct")
    api_key = kwargs.get("api_key", "dummy-key-for-vllm")
    
    # Create vLLM clients for each endpoint
    vllm_clients = [
        openai_llm(
            provider="vllm",
            base_url=url,
            model_name=model_name,
            api_key=api_key
        ) for url in base_urls
    ]
    
    # Create messages for each answer that needs verification
    message_list = []
    for ind in idxs:
        message = get_compare_messages(question_strs[ind], answer_strs[ind], ground_truths[ind])
        message_list.append(message)
    
    # Distribute the messages across the available vLLM clients
    num_clients = len(vllm_clients)
    
    async def process_messages_in_parallel():
        # Split messages among clients
        message_chunks = [[] for _ in range(num_clients)]
        client_to_orig_idx = {}  # Maps (client_idx, chunk_position) to original message index
        
        # Distribute messages across clients
        for i, msg in enumerate(message_list):
            client_idx = i % num_clients
            chunk_pos = len(message_chunks[client_idx])
            message_chunks[client_idx].append(msg)
            client_to_orig_idx[(client_idx, chunk_pos)] = i
        
        # Process chunks in parallel
        tasks = []
        for i, client in enumerate(vllm_clients):
            if message_chunks[i]:  # Only create tasks for chunks with messages
                tasks.append(client.generate_outputs_async(message_chunks[i]))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Create output array with correct size
        vllm_outputs = ["<judge>1</judge>"] * len(message_list)
        
        # Map results back to their original positions
        for client_idx, client_results in enumerate(results):
            for chunk_pos, result in enumerate(client_results):
                orig_idx = client_to_orig_idx.get((client_idx, chunk_pos))
                vllm_outputs[orig_idx] = result
                
        # Give a short delay to ensure all HTTP connections can complete properly
        await asyncio.sleep(1)
        
        return vllm_outputs
    
    # Run parallel processing
    vllm_outputs = asyncio.run(process_messages_in_parallel())
    
    vllm_corrects_num = 0
    
    # Process the outputs
    for i, vllm_output in enumerate(vllm_outputs):
        # Find the content between <judge> and </judge> tags
        judge_match = re.search(r"<judge>(.*?)</judge>", vllm_output, re.DOTALL)
        if judge_match:
            judge_content = judge_match.group()
        else:
            judge_content = "<judge>1</judge>"
            
        if judge_content.strip() == "<judge>0</judge>":  # Judge as correct
            acc_rewards[idxs[i]] = 1.0
            vllm_corrects_num += 1
            
    print(f"Corrected {vllm_corrects_num}/{len(idxs)} rewards using vLLM as judge with {num_clients} endpoints.")
    return acc_rewards

def accuracy_reward_batch_wo_LMM(predict_strs, ground_truths, prompt_strs, response_length):
    acc_rewards = []
    for predict_str, ground_truth in zip(predict_strs, ground_truths):
        acc_reward = r1v_accuracy_reward(predict_str, ground_truth, response_length)
        acc_rewards.append(acc_reward)
    return acc_rewards

def openr1_compute_score_batch(predict_strs: list, ground_truths: list, prompt_strs: list, validation: bool = False, response_length = None, cos_len_reward_config: list = None) -> float:
    """Compute reward score based on the completion and ground truth.

    Args:
        predict_str: The completion string
        ground_truth: The ground truth string
    """
    # breakpoint()
    acc_rewards = accuracy_reward_batch_w_LMM_as_judge(predict_strs, ground_truths, prompt_strs, response_length)
    format_rewards = format_reward_batch(predict_strs)
    cosine_len_rewards = get_cosine_scaled_reward_batch(min_value_wrong=cos_len_reward_config['min_value_wrong'], max_value_wrong=cos_len_reward_config['max_value_wrong'], min_value_correct=cos_len_reward_config['min_value_correct'], max_value_correct=cos_len_reward_config['max_value_correct'], max_len=response_length)(predict_strs, ground_truths, acc_rewards)
    repetition_penalty_rewards = get_repetition_penalty_reward()(predict_strs)
    # if validation:
    #     rewards = acc_rewards
    # else:
    #     rewards = [acc_reward + format_reward + cosine_len_reward + repetition_penalty_reward for acc_reward, format_reward, cosine_len_reward, repetition_penalty_reward in zip(acc_rewards, format_rewards, cosine_len_rewards, repetition_penalty_rewards)]
    # return rewards
    reward_dicts = []
    for acc_reward, format_reward, cosine_len_reward, repetition_penalty_reward in zip(acc_rewards, format_rewards, cosine_len_rewards, repetition_penalty_rewards):
        reward_dict = {
            "overall": acc_reward + format_reward + cosine_len_reward + repetition_penalty_reward,
            "accuracy": acc_reward,
            "format": format_reward,
            "cosine_len": cosine_len_reward,
            "repetition_penalty": repetition_penalty_reward
        }
        reward_dicts.append(reward_dict)
    return reward_dicts

def openr1_compute_score_batch_wo_LMM(predict_strs: list, ground_truths: list, prompt_strs: list, validation: bool = False, response_length = None, cos_len_reward_config: list = None) -> float:
    """Compute reward score based on the completion and ground truth.

    Args:
        predict_str: The completion string
        ground_truth: The ground truth string
    """
    # breakpoint()
    acc_rewards = accuracy_reward_batch_wo_LMM(predict_strs, ground_truths, prompt_strs, response_length)
    format_rewards = format_reward_batch(predict_strs)
    cosine_len_rewards = get_cosine_scaled_reward_batch(min_value_wrong=cos_len_reward_config['min_value_wrong'], max_value_wrong=cos_len_reward_config['max_value_wrong'], min_value_correct=cos_len_reward_config['min_value_correct'], max_value_correct=cos_len_reward_config['max_value_correct'], max_len=response_length)(predict_strs, ground_truths, acc_rewards)    
    repetition_penalty_rewards = get_repetition_penalty_reward()(predict_strs)
    reward_dicts = []
    for acc_reward, format_reward, cosine_len_reward, repetition_penalty_reward in zip(acc_rewards, format_rewards, cosine_len_rewards, repetition_penalty_rewards):
        reward_dict = {
            "overall": acc_reward + format_reward + cosine_len_reward + repetition_penalty_reward,
            "accuracy": acc_reward,
            "format": format_reward,
            "cosine_len": cosine_len_reward,
            "repetition_penalty": repetition_penalty_reward
        }
        reward_dicts.append(reward_dict)
    return reward_dicts

def openr1_compute_score_batch_vllm(predict_strs: list, ground_truths: list, prompt_strs: list, validation: bool = False, response_length = None, cos_len_reward_config: list = None, **kwargs) -> float:
    """Compute reward score based on the completion and ground truth.

    Args:
        predict_str: The completion string
        ground_truth: The ground truth string
        **kwargs: Additional arguments for vLLM client:
            base_urls: List of URLs for vLLM servers for parallel processing
            base_url: Single URL for vLLM server (deprecated, use base_urls instead)
            model_name: The name of the model to use
            api_key: API key for the vLLM server (can be dummy for vLLM)
    """
    acc_rewards = accuracy_reward_batch_vllm(predict_strs, ground_truths, prompt_strs, response_length, **kwargs)
    format_rewards = format_reward_batch(predict_strs)
    cosine_len_rewards = get_cosine_scaled_reward_batch(min_value_wrong=cos_len_reward_config['min_value_wrong'], max_value_wrong=cos_len_reward_config['max_value_wrong'], min_value_correct=cos_len_reward_config['min_value_correct'], max_value_correct=cos_len_reward_config['max_value_correct'], max_len=response_length)(predict_strs, ground_truths, acc_rewards)
    repetition_penalty_rewards = get_repetition_penalty_reward()(predict_strs)
    reward_dicts = []
    for acc_reward, format_reward, cosine_len_reward, repetition_penalty_reward in zip(acc_rewards, format_rewards, cosine_len_rewards, repetition_penalty_rewards):
        reward_dict = {
            "overall": acc_reward + format_reward + cosine_len_reward + repetition_penalty_reward,
            "accuracy": acc_reward,
            "format": format_reward,
            "cosine_len": cosine_len_reward,
            "repetition_penalty": repetition_penalty_reward
        }
        reward_dicts.append(reward_dict)
    return reward_dicts