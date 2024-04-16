from fastapi import FastAPI
from pydantic import BaseModel, Extra
import argparse
from typing import Optional
from transformers import set_seed
import httpx
import random


class Prompt(BaseModel, extra=Extra.allow):
    prompt: str
    seed: Optional[int] = 0
    max_length: Optional[int] = 77


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--netuid", type=str, default=23)
    parser.add_argument("--min_stake", type=int, default=100)
    parser.add_argument(
        "--chain_endpoint",
        type=str,
        default="finney",
    )
    parser.add_argument("--disable_secure", action="store_true", default=False)
    args = parser.parse_args()
    return args


class ChallengeImage:
    def __init__(self):
        self.app = FastAPI()
        self.app.add_api_route("/", self.__call__, methods=["POST"])

    async def __call__(
        self,
        data: Prompt,
    ):
        data = dict(data)
        seed = random.randint(0, 1e9)
        set_seed(seed)
        prompt = data["prompt"]
        if not prompt:
            prompt = "an image of "
        async with httpx.AsyncClient() as httpx_client:
            response = await httpx_client.post(
                "http://localhost:8000/v1/completions",
                json={
                    "prompt": [prompt],
                    "model": "LykosAI/GPT-Prompt-Expansion-Fooocus-v2",
                    "max_tokens": 77,
                },
            )
        prompt_completion = response.json()["choices"][0]["text"].strip()
        prompt = prompt + prompt_completion
        return {"prompt": prompt}


app = ChallengeImage()
