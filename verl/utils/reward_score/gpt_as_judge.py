import json
import math
from collections import defaultdict, Counter
from openai import AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)
import re
import os
import json

from tqdm.asyncio import tqdm_asyncio
import asyncio
import jieba

from math_verify import parse, verify
from mathruler.grader import _is_float, _is_int, _str_is_int, _str_to_int, _parse_latex, _inject_implicit_mixed_number,extract_boxed_content, grade_answer

from dotenv import load_dotenv



def get_compare_messages(question,response,answer):
    prompt = f"""
Your task is to determine whether the user's answer is correct based on the provided questions and standard answers (for example, if the user expresses a similar meaning to the standard answer, or another interpretation of the standard answer, it is considered correct.)

Note(very important!):
1.If the standard answer is an interval, and the user's answer is only one value, it is considered wrong. If the standard answer has only one option, and the user's answer has multiple options, it is also considered wrong.
2. If the user's answer has no unit, but the value is consistent with the standard answer, it is also considered correct, such as 100 and 100m, 25t and 25,they are considered to be the same.
3. If the answer is an equation and the answer is just a value, it is OK as long as the value is consistent with the value after the equation, such as area=pi/25\xa0^2 is same as pi/25.

The question is: {question}

The standard answer: {answer}

The user's answer: {response}

Please strictly follow the following format for output(0 represents correct, 1 represents incorrect):
<think>{{your concise think step}}</think>
<judge>{{0/1}}</judge>

for example:
<think>The standard answer is right, and the user's answer is right frontal lobe, they express the same meaning, so it is correct.</think>
<judge>0</judge>

<think>The standard answer is 0.5, and the user's answer is \\frac{{1}}{{2}}. The numerical calculations of the two are consistent, so it is correct.</think>
<judge>0</judge>

<think>The standard answer is 0.5, and the user's answer is \\frac{{1}}{{3}}. The numerical calculations of the two are inconsistent, so it is incorrect.</think>
<judge>1</judge>

<think>The standard answer is 2t, and the user's answer is 2. The value before the unit is the same, so it is correct.</think>
<judge>0</judge>
    """
    messages = [{"role":"user","content":prompt}]
    return messages


class fake_response:
    def __init__(self,usage):
        self.usage = usage

def before_retry_fn(retry_state):
    if retry_state.attempt_number > 1:
        print(f"Retrying API call. Attempt #{retry_state.attempt_number}, f{retry_state}")

async def deal_tasks(tasks, max_concurrent_tasks=256):
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    results = []

    async def sem_task(task):
        async with semaphore:
            return await task  # 注意这里是调用task()

    # 创建未执行的协程列表
    sem_tasks = [sem_task(task) for task in tasks]

    # 使用tqdm_asyncio.gather来调度任务并显示进度
    print("Calling model to verify answers...")
    for coro in tqdm_asyncio.as_completed(sem_tasks, total=len(sem_tasks)):
        result = await coro
        results.append(result)

    return results


class openai_llm:
    def __init__(self, provider="azure", **kwargs):
        """
        Initialize the LLM client.
        
        Args:
            provider (str): The provider type - "azure" or "vllm"
            **kwargs: Additional arguments for specific providers
                For vllm: base_url, api_key, model_name are required
                For azure: Uses environment variables by default
        """
        load_dotenv()
        
        self.provider = provider
        self.token_log_file = "./logs/token.json"
        
        if provider == "azure":
            # Azure OpenAI settings
            self.endpoint = os.getenv("OPENAI_ENDPOINT")
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.api_version = "2024-02-15-preview"
            self.deployment = "gpt-4o-mini-0718"
            self.model = "gpt-4o-mini-0718"
            
            self.client = AzureOpenAI(
                azure_deployment=self.deployment,
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version
            )
            self.async_client = AsyncAzureOpenAI(
                azure_deployment=self.deployment,
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version
            )
        elif provider == "vllm":
            # vLLM with OpenAI-compatible API settings
            self.base_url = kwargs.get("base_url", "http://localhost:8000/v1")
            self.api_key = kwargs.get("api_key", None)  # Make API key optional
            self.model = kwargs.get("model_name", "NousResearch/Meta-Llama-3-8B-Instruct")
            
            # Initialize client without api_key if it's None
            if self.api_key is None:
                self.client = OpenAI(
                    base_url=self.base_url
                )
                self.async_client = AsyncOpenAI(
                    base_url=self.base_url
                )
            else:
                self.client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key
                )
                self.async_client = AsyncOpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key
                )
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'azure' or 'vllm'")
    
    def cal_cost(self,response,**kwargs):
        if not os.path.exists(self.token_log_file):
            with open(self.token_log_file, "w") as f:
                json.dump({"none":"none"},f)
        with open(self.token_log_file, "r") as f:
            tokens = json.load(f)
            current_model = kwargs.get("model", self.model)
            if current_model not in tokens:
                tokens[current_model] = [0,0]
            tokens[current_model][0] += response.usage.prompt_tokens
            tokens[current_model][1] += response.usage.completion_tokens
        with open(self.token_log_file, "w") as f:
            json.dump(tokens, f)
    
    def cal_batch_cost(self,prompt_tokens,completion_tokens,**kwargs):
        if not os.path.exists(self.token_log_file):
            with open(self.token_log_file, "w") as f:
                json.dump({"none":"none"},f)
        with open(self.token_log_file, "r") as f:
            tokens = json.load(f)
            current_model = kwargs.get("model", self.model)
            if current_model not in tokens:
                tokens[current_model] = [0,0]
            tokens[current_model][0] += prompt_tokens
            tokens[current_model][1] += completion_tokens
        with open(self.token_log_file, "w") as f:
            json.dump(tokens, f)

    @retry(wait=wait_fixed(3), stop=stop_after_attempt(5), before=before_retry_fn)
    def response(self,messages,**kwargs):
        model = kwargs.get("model", self.model)
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            n=kwargs.get("n", 1),
            temperature=kwargs.get("temperature", 0),
            max_tokens=kwargs.get("max_tokens", 4000),
            timeout=kwargs.get("timeout", 180)
        )
        # self.cal_cost(response,**kwargs)
        return response.choices[0].message.content
    
    @retry(wait=wait_fixed(3), stop=stop_after_attempt(5), before=before_retry_fn)
    async def response_async(self,messages,**kwargs):
        model = kwargs.get("model", self.model)
        
        response = await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
            n=kwargs.get("n", 1),
            temperature=kwargs.get("temperature", 0),
            max_tokens=kwargs.get("max_tokens", 4096),
            timeout=kwargs.get("timeout", 180)
        )      
        # self.cal_cost(response,**kwargs)
        return response.choices[0].message.content
    
    def generate_output(self,messages,**kwargs):
        try:
            response = self.response(messages,**kwargs)
        except Exception as e:
            response = "<judge>1</judge>" # if failed, return not match
            print(f"get {kwargs.get('model', self.model)} response failed: {e}")
        return response
    
    async def generate_output_async(self,idx, messages,**kwargs):
        try:
            response = await self.response_async(messages,**kwargs)
        except Exception as e:
            response = "<judge>1</judge>" # if failed, return not match
            print(f"get {kwargs.get('model', self.model)} response failed: {e}")
        return idx,response
    
    def generate_outputs(self,messages,**kwargs):
        tasks = [self.generate_output_async(i,messages[i],**kwargs) for i in range(len(messages))]
        results = asyncio.run(deal_tasks(tasks))
        results = sorted(results, key=lambda x: x[0])
        results = [x[1] for x in results]
        return results

# Default instance using Azure OpenAI
# judger = openai_llm()

if __name__ == "__main__":
    
    # Test with Azure OpenAI
    messages = [[{"role":"user","content":"how are you"}] for _ in range(3)]
    llm = openai_llm()
    results = llm.generate_outputs(messages)
    print("Azure OpenAI results:", results)
    
    # Test with vLLM
    try:
        llm_vllm = openai_llm(
            provider="vllm",
            base_url="http://localhost:8000/v1",  # Change this to the actual vLLM server address
            model_name="NousResearch/Meta-Llama-3-8B-Instruct"  # Change this to the actual model name
            # No API key needed when server doesn't require authentication
        )
        vllm_messages = [[{"role":"user","content":"Tell me a short joke"}]]
        vllm_results = llm_vllm.generate_outputs(vllm_messages)
        print("vLLM results:", vllm_results)
    except Exception as e:
        print(f"vLLM test failed (this is expected if vLLM server is not running): {e}")

