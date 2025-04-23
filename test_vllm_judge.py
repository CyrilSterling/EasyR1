#!/usr/bin/env python3
"""
Test script for vLLM model as judge for reward calculation
"""

import os
import sys
from typing import List, Dict, Any

# Set the API key for OpenAI client
os.environ["OPENAI_API_KEY"] = "dummy-key-for-vllm"

# Add the EasyR1 root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the necessary functions
from verl.utils.reward_score.openr1_rewards_batch import openr1_compute_score_batch_vllm
from verl.utils.reward_score.gpt_as_judge import openai_llm, get_compare_messages

def test_vllm_judge(base_url: str = "http://localhost:8000/v1", 
                  model_name: str = "Qwen2.5-1.5B-Instruct",
                  api_key: str = "dummy-key-for-vllm"):
    """
    Test the vLLM model as judge functionality.
    
    Args:
        base_url: URL of the vLLM server
        model_name: Name of the model
        api_key: API key for the vLLM server (can be a dummy value)
    """
    print(f"Testing vLLM judge with model {model_name} at {base_url}")
    
    # Test data
    predict_strs = [
        "<think>\nTo find the area of a rectangle, I need to multiply the width by the height.\nWidth = 6 units\nHeight = 7 units\nArea = 6 × 7 = 42 square units\n</think>\n<answer>\n42 square units\n</answer>",
        "<think>\nTo find the area of a rectangle, I need to multiply the width by the height.\nWidth = 6 units\nHeight = 7 units\nArea = 6 × 7 = 43 square units\n</think>\n<answer>\n43 square units\n</answer>",
        "<think>\nI need to calculate the value of 3^4.\n3^4 = 3 × 3 × 3 × 3 = 9 × 9 = 81\n</think>\n<answer>\n81\n</answer>"
    ]
    
    ground_truths = [
        "45 square units",
        "42 square units",
        "81"
    ]
    
    prompt_strs = [
        "Here's a math problem:\nuser\nFind the area of a rectangle with width 6 and height 7.\nassistant\n",
        "Here's a math problem:\nuser\nFind the area of a rectangle with width 6 and height 7.\nassistant\n",
        "Here's a math problem:\nuser\nCalculate 3^4.\nassistant\n"
    ]
    
    response_length = 2048
    
    # Set up configuration for reward calculation
    cos_len_reward_config = {
        'min_value_wrong': -1.0,
        'max_value_wrong': -0.5,
        'min_value_correct': 0.0,
        'max_value_correct': 0.0
    }
    
    # Insert a breakpoint here for debugging
    print("About to call openr1_compute_score_batch_vllm. Add breakpoint here if needed.")
    
    # Calculate rewards using vLLM
    rewards = openr1_compute_score_batch_vllm(
        predict_strs=predict_strs,
        ground_truths=ground_truths,
        prompt_strs=prompt_strs,
        validation=False,
        response_length=response_length,
        cos_len_reward_config=cos_len_reward_config,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key  # Pass the API key
    )
    
    # Print results
    print("\nReward results:")
    for i, (pred, truth, reward) in enumerate(zip(predict_strs, ground_truths, rewards)):
        print(f"\nExample {i+1}:")
        print(f"Prediction: {pred.split('<answer>')[1].split('</answer>')[0].strip()}")
        print(f"Ground truth: {truth}")
        print(f"Accuracy reward: {reward['accuracy']}")
        print(f"Format reward: {reward['format']}")
        print(f"Overall reward: {reward['overall']}")
    
    # Test direct API call to the judge
    print("\nTesting direct vLLM judge API call:")
    
    vllm_client = openai_llm(
        provider="vllm",
        base_url=base_url,
        model_name=model_name,
        api_key=api_key  # Pass the API key
    )
    
    # Create a test message
    test_message = get_compare_messages(
        question="Find the area of a rectangle with width 6 and height 7.",
        response="42 square units",
        answer="42 square units"
    )
    
    # Call the vLLM judge directly
    response = vllm_client.generate_output(test_message)
    print(f"\nDirect judge response: {response}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test vLLM judge functionality")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1", 
                      help="URL of the vLLM server")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-1.5B-Instruct",
                      help="Name of the model")
    parser.add_argument("--api_key", type=str, default="dummy-key-for-vllm",
                      help="API key for the vLLM server (can be a dummy value)")
    
    args = parser.parse_args()
    
    test_vllm_judge(base_url=args.base_url, model_name=args.model_name, api_key=args.api_key) 