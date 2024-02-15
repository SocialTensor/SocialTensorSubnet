from fastapi import FastAPI, Request, Response
import bittensor as bt
import torch
from typing import List, Union
from services.rewarding.utils import instantiate_from_config, measure_time
from services.rewarding.hash_compare import infer_hash
from pydantic import BaseModel
import uvicorn
import argparse
import time
import threading
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
import yaml
import argparse
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

MODEL_CONFIG = yaml.load(open("configs/model_config.yaml"), yaml.FullLoader)


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
        "--category", type=str, default="TextToImage", choices=list(MODEL_CONFIG.keys())
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="RealisticVision",
    )
    args = parser.parse_args()
    if args.model_name not in MODEL_CONFIG[args.category]:
        raise ValueError(
            (
                f"Model name {args.model_name} not found in category {args.category}"
                f"Available models are {list(MODEL_CONFIG[args.category].keys())}"
            )
        )
    return args


class Prompt(BaseModel):
    prompt: str
    seed: int
    image: str
    pipeline_params: dict = {}
    conditional_image: str = ""


class RewardRequest(BaseModel):
    miner_data: List[Prompt]
    base_data: Prompt


app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ARGS = get_args()
MODEL = instantiate_from_config(MODEL_CONFIG[ARGS.category][ARGS.model_name])


@app.middleware("http")
@limiter.limit("60/minute")
async def filter_allowed_ips(request: Request, call_next):
    if ARGS.disable_secure:
        response = await call_next(request)
        return response
    if (request.client.host not in ALLOWED_IPS) and (
        request.client.host != "127.0.0.1"
    ):
        print(f"Blocking an unallowed ip:", request.client.host, flush=True)
        return Response(
            content="You do not have permission to access this resource",
            status_code=403,
        )
    print(f"Allow an ip:", request.client.host, flush=True)
    response = await call_next(request)
    return response


@app.post("/")
async def get_rewards(
    reward_request: RewardRequest,
):
    base_data = reward_request.base_data
    miner_data = reward_request.miner_data
    generator = torch.manual_seed(base_data.seed)
    validator_result = MODEL(
        generator=generator,
        **base_data.pipeline_params,
        **base_data.dict(),
    )
    validator_image = validator_result.images[0]
    miner_images = [d.image for d in miner_data]
    rewards = infer_hash(validator_image, miner_images)
    rewards = [float(reward) for reward in rewards]
    return {
        "rewards": rewards,
    }


def define_allowed_ips(url, netuid, min_stake):
    global ALLOWED_IPS
    ALLOWED_IPS = []
    while True:
        all_allowed_ips = []
        subtensor = bt.subtensor(url)
        metagraph = subtensor.metagraph(netuid)
        for uid in range(len(metagraph.total_stake)):
            if metagraph.total_stake[uid] > min_stake:
                all_allowed_ips.append(metagraph.axons[uid].ip)
        ALLOWED_IPS = all_allowed_ips
        print("Updated allowed ips:", ALLOWED_IPS, flush=True)
        time.sleep(60)


if __name__ == "__main__":
    if not ARGS.disable_secure:
        allowed_ips_thread = threading.Thread(
            target=define_allowed_ips,
            args=(
                ARGS.chain_endpoint,
                ARGS.netuid,
                ARGS.min_stake,
            ),
        )
        allowed_ips_thread.setDaemon(True)
        allowed_ips_thread.start()
    uvicorn.run(app, host="0.0.0.0", port=ARGS.port)
