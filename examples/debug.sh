set -x

# pip install torchdata math-verify latex2sympy2_extended antlr4-python3-runtime==4.9.3 asyncio openai tenacity

# pip install tensorboard (04.01 merge)

SYSTEM_PROMPT="""A conversation between User and Assistant. The User provides an image and asks a question. The Assistant first analyzes both the image and the question, then carefully thinks about the reasoning process step by step, and finally provides the User with an accurate answer. The Assistant must carefully checkout the correctness and validity of each reasoning step. If any errors or inconsistencies are found during the reasoning process, the Assistant reflects and corrects them logically. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here, with potential reflections and corrections </think><answer> final answer here, with the key result enclosed in \boxed{} </answer>."""

python3 -m verl.trainer.main \
    config=examples/debug.yaml \
    data.system_prompt="${SYSTEM_PROMPT}" \

