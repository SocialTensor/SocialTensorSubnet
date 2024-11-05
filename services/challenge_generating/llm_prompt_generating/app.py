import argparse
from fastapi import FastAPI
import uvicorn
import random
from transformers import AutoTokenizer
from prometheus_fastapi_instrumentator import Instrumentator


def get_args():
    parser = argparse.ArgumentParser(
        description="Run the LLM Prompt Generating service"
    )
    parser.add_argument(
        "--port", type=int, default=10001, help="Port to run the service on"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of workers to run the service on"
    )
    args = parser.parse_args()
    return args


class LLMPromptGenerating:
    def __init__(self):
        self.app = FastAPI()
        self.tokenizer = AutoTokenizer.from_pretrained("casperhansen/llama-3-70b-instruct-awq")
        self.app.add_api_route("/", self.generate, methods=["POST"])
        with open("services/challenge_generating/questions.txt") as f:
            self.questions = f.readlines()
        self.questions = [q.strip() for q in self.questions]
        Instrumentator().instrument(self.app).expose(self.app)
    async def generate(self):
        try:
            prompts = random.choices(self.questions, k=20)
            prompt = "\n".join(prompts)
        except:
            prompt = "Tell me an event that happened in the history of the world."
        print(prompt, flush=True)
        conversation = [
            {"role": "user", "content": prompt},
        ]
        chat_prompt = self.tokenizer.apply_chat_template(
            conversation, tokenize=False
        )
        data = {
            "prompt_input": chat_prompt,
            "pipeline_params": {
                "max_tokens": 512,
                "min_tokens": 512,
                "logprobs": 100,
                "stop": ["<|eot_id|>"],
            },
        }
        return data

llm_prompt_generating = LLMPromptGenerating()
