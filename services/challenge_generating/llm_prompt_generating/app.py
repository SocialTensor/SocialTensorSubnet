import argparse
from fastapi import FastAPI
import uvicorn
from services.challenge_generating.llm_prompt_generating.twitter_prompt import (
    TwitterPrompt,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Run the LLM Prompt Generating service"
    )
    parser.add_argument(
        "--port", type=int, default=10001, help="Port to run the service on"
    )
    args = parser.parse_args()
    return args


class LLMPromptGenerating:
    def __init__(self, args):
        self.app = FastAPI()
        self.twitter_prompt = TwitterPrompt(max_tokens=1024)
        self.args = args
        self.app.add_api_route("/", self.generate, methods=["POST"])

    async def generate(self):
        try:
            prompt = self.get_twitter_prompt()
        except:
            prompt = "Tell me an event that happened in the history of the world."
        data = {
            "prompt": prompt,
            "pipeline_params": {
                "max_tokens": 4096,
                "logprobs": 100,
            },
        }
        return data

    def get_twitter_prompt(self):
        prompt = self.twitter_prompt()
        return prompt


if __name__ == "__main__":
    args = get_args()
    llm_prompt_generating = LLMPromptGenerating(args)

    uvicorn.run(llm_prompt_generating.app, host="0.0.0.0", port=args.port)
