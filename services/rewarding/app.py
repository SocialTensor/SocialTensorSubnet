from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
import uvicorn
import argparse
import threading
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
import yaml
from services.rays.image_generating import ModelDeployment
from ray import serve
from ray.serve.handle import DeploymentHandle
from functools import partial
from services.rewarding.cosine_similarity_compare import CosineSimilarityReward
from services.rewarding.open_category_reward import OpenCategoryReward
import asyncio
from services.owner_api_core import define_allowed_ips, filter_allowed_ips, limiter
from prometheus_fastapi_instrumentator import Instrumentator
import diskcache as dc

MODEL_CONFIG = yaml.load(
    open("generation_models/configs/model_config.yaml"), yaml.FullLoader
)


class RequestCancelledMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Let's make a shared queue for the request messages
        queue = asyncio.Queue()

        async def message_poller(sentinel, handler_task):
            nonlocal queue
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    handler_task.cancel()
                    return sentinel  # Break the loop

                # Puts the message in the queue
                await queue.put(message)

        sentinel = object()
        handler_task = asyncio.create_task(self.app(scope, queue.get, send))
        asyncio.create_task(message_poller(sentinel, handler_task))

        try:
            return await handler_task
        except asyncio.CancelledError:
            print("Cancelling request due to disconnect")


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
    parser.add_argument("--disable_secure", action="store_true")
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
    )
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
    parser.add_argument(
        "--webhook_url",
        type=str,
        default="",
    )
    parser.add_argument(
        "--notice_prob",
        type=float,
        default=0.2,
    )

    args = parser.parse_args()
    return args


class Prompt(BaseModel):
    prompt: str
    seed: int
    image: str
    pipeline_type: str
    pipeline_params: dict = {}
    conditional_image: str = ""


class RewardRequest(BaseModel):
    miner_data: List[Prompt]
    base_data: Prompt


class BaseRewardApp:
    def __init__(self, args):
        self.args = args
        self.app = FastAPI()
        self.app.add_api_route("/", self.__call__, methods=["POST"])
        self.app.middleware("http")(partial(filter_allowed_ips, self))
        self.app.state.limiter = limiter
        self.app.add_middleware(RequestCancelledMiddleware)
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        Instrumentator().instrument(self.app).expose(self.app)
        self.allowed_ips = []

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

    async def __call__(self, reward_request: RewardRequest):
        raise NotImplementedError("This method should be implemented by subclasses")

class FixedCategoryRewardApp(BaseRewardApp):
    def __init__(self, model_handle: DeploymentHandle, args):
        super().__init__(args)
        self.rewarder = CosineSimilarityReward()
        self.model_handle = model_handle
        self.cache = dc.Cache("reward_app_cache")
        self.ttl = 600
        self.prompt_cache = dc.Cache("prompt_cache")
        self.reward_threshold_for_check_cache = 0.5
        self.cosine_similarity_threshold_for_check_cache = 0.5

    async def __call__(self, reward_request: RewardRequest):
        base_data = reward_request.base_data
        miner_data = reward_request.miner_data
        validator_image = self.cache.get((base_data.prompt, base_data.seed))
        if validator_image is None:
            validator_image = await self.model_handle.generate.remote(prompt_data=base_data)
            self.cache.set((base_data.prompt, base_data.seed), validator_image, expire=self.ttl)

            another_seed_of_this_prompt = self.prompt_cache.get(base_data.prompt)
            if another_seed_of_this_prompt is None:
                self.prompt_cache.set((base_data.prompt), [base_data.seed], expire=self.ttl)
            else:
                self.prompt_cache.set((base_data.prompt), another_seed_of_this_prompt + [base_data.seed], expire=self.ttl)

        miner_images = [d.image for d in miner_data]
        rewards = self.rewarder.get_reward(
            validator_image, miner_images, base_data.pipeline_type
        )
        rewards = [float(reward) for reward in rewards]

        rewards = self.verify_miner_images(miner_images, rewards, base_data)
        print(rewards, flush=True)
        return {"rewards": rewards}
    
    def verify_miner_images(self, miner_images, rewards, base_data):
        for i, miner_image in enumerate(miner_images):
            if rewards[i] < self.reward_threshold_for_check_cache:
                another_seed_of_this_prompt = self.prompt_cache.get(base_data.prompt)
                if another_seed_of_this_prompt is None:
                    continue

                validator_images_for_this_prompt = [
                    self.cache.get((base_data.prompt, seed))
                    for seed in another_seed_of_this_prompt
                    if seed != base_data.seed 
                        and self.cache.get((base_data.prompt, seed)) is not None
                ]
                cosine_similarity_scores = self.rewarder.get_cosine_similarity(miner_image, validator_images_for_this_prompt)
                if cosine_similarity_scores[i] > self.cosine_similarity_threshold_for_check_cache:
                    rewards[i] = -1.0
                    
        return rewards

class OpenCategoryRewardApp(BaseRewardApp):
    def __init__(self, args):
        super().__init__(args)
        self.rewarder = OpenCategoryReward()

    async def __call__(self, reward_request: RewardRequest):
        base_data = reward_request.base_data
        miner_data = reward_request.miner_data
        rewards = self.rewarder.get_reward(
            base_data.prompt, [x.image for x in miner_data]
        )
        rewards = [float(reward) for reward in rewards]
        print(rewards, flush=True)
        return {"rewards": rewards}


if __name__ == "__main__":
    args = get_args()
    if args.model_name:
        print("Starting fixed category reward app", flush=True)
        model_deployment = serve.deployment(
            ModelDeployment,
            name="model_deployment",
            num_replicas=args.num_replicas,
            ray_actor_options={"num_gpus": args.num_gpus},
        )
        serve.run(
            model_deployment.bind(
                MODEL_CONFIG[args.model_name],
            ),
            name="model_deployment",
        )
        model_handle = serve.get_deployment_handle(
            "model_deployment", "model_deployment"
        )
        app = FixedCategoryRewardApp(model_handle, args)
    else:
        print("Starting open category reward app", flush=True)
        app = OpenCategoryRewardApp(args)
    uvicorn.run(
        app.app,
        host="0.0.0.0",
        port=args.port,
    )
