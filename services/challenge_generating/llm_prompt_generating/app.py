import argparse
from fastapi import FastAPI
import uvicorn
from services.challenge_generating.llm_prompt_generating.random_text_seed import (
    get_random_seeds,
)
from transformers import AutoTokenizer
import httpx


def get_args():
    parser = argparse.ArgumentParser(
        description="Run the LLM Prompt Generating service"
    )
    parser.add_argument(
        "--port", type=int, default=10001, help="Port to run the service on"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-7b-it",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--vllm_url",
        type=str,
        default="http://localhost:8000",
        help="URL of the VLLM service",
    )
    args = parser.parse_args()
    return args


class LLMPromptGenerating:
    def __init__(self, args):
        self.app = FastAPI()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.args = args
        self.app.add_api_route("/", self.generate, methods=["GET"])

    async def generate(self):
        prompt = self.get_question_prompt()
        data = {
            "model": self.args.model_name,
            "prompt": [prompt],
            "max_tokens": 512,
        }
        response = httpx.post(f"{self.args.vllm_url}/v1/completions", json=data)
        response.raise_for_status()
        response = response.json()
        return response["choices"][0]["text"].strip()

    def get_question_prompt(self):
        chat = [
            {
                "role": "user",
                "content": f"Your task is to write a question that will be used to evaluate how intelligent different AI models are. The question should be fairly complex and it can be about anything as long as it somehow relates to these topics: {get_random_seeds()}",
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        prompt = prompt + "Question:"
        return prompt


if __name__ == "__main__":
    args = get_args()
    llm_prompt_generating = LLMPromptGenerating(args)

    uvicorn.run(llm_prompt_generating.app, host="0.0.0.0", port=args.port)
