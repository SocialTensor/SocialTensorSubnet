from fastapi import FastAPI
from pydantic import BaseModel, Extra
import uvicorn
import argparse
import threading
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from typing import Optional
import re
from ray import serve
from ray.serve.handle import DeploymentHandle
from transformers import pipeline, set_seed
import random
from functools import partial
from services.owner_api_core import define_allowed_ips, filter_allowed_ips, limiter

with open("services/challenge_generating/prompt_generating/idea.txt", "r") as file:
    ideas = file.readlines()


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
    parser.add_argument(
        "--num_gpus",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--num_replicas",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    return args


class PromptGenerator:
    def __init__(self):
        generator = pipeline(
            "text-generation",
            model="Gustavosta/MagicPrompt-Stable-Diffusion",
            device="cuda",
            tokenizer="gpt2",
        )
        self.generator = generator

    async def __call__(self, data: dict):
        set_seed(data["seed"])
        if not data["prompt"]:
            starting_text = "" if random.random() < 0.5 else "image of"
        else:
            starting_text = data["prompt"]
        prompt = self.generate(starting_text)
        prompt = prompt.replace("image of", "")
        prompt = prompt.strip()
        print("Prompt Generated:", prompt, flush=True)
        return prompt

    def generate(self, starting_text):
        if starting_text == "":
            starting_text: str = (
                ideas[random.randrange(0, len(ideas))]
                .replace("\n", "")
                .lower()
                .capitalize()
            )
            starting_text: str = re.sub(r"[,:\-–.!;?_]", "", starting_text)

        response = self.generator(
            starting_text,
            max_length=(len(starting_text) + random.randint(60, 90)),
            num_return_sequences=1,
        )
        prompt = response[0]["generated_text"].strip()
        return prompt


class ChallengeImage:
    def __init__(self, model_handle: DeploymentHandle, args):
        self.args = args
        self.model_handle = model_handle
        self.app = FastAPI()
        self.app.add_api_route("/", self.__call__, methods=["POST"])
        self.app.middleware("http")(partial(filter_allowed_ips, self))
        self.app.state.limiter = limiter
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        if not self.args.disable_secure:
            self.allowed_ips_thread = threading.Thread(
                target=define_allowed_ips,
                args=(
                    self,
                    self.args.chain_endpoint,
                    self.args.netuid,
                    self.args.min_stake,
                ),
            )
            self.allowed_ips_thread.daemon = True
            self.allowed_ips_thread.start()

    async def __call__(
        self,
        data: Prompt,
    ):
        data = dict(data)
        prompt = await self.model_handle.remote(data)
        return {"prompt": prompt}


if __name__ == "__main__":
    args = get_args()
    print(args)
    model_deployment = serve.deployment(
        PromptGenerator,
        name="deployment",
        num_replicas=args.num_replicas,
        ray_actor_options={"num_gpus": args.num_gpus},
    )
    serve.run(
        model_deployment.bind(),
        name="deployment-prompt-challenge",
    )
    model_handle = serve.get_deployment_handle(
        "deployment", "deployment-prompt-challenge"
    )
    app = ChallengeImage(model_handle, args)
    uvicorn.run(app.app, host="0.0.0.0", port=args.port)
