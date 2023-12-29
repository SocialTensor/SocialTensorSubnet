from transformers import pipeline
from fastapi import FastAPI, Request, Response, Depends
import torch
import numpy as np
import bittensor as bt
import random
from pydantic import BaseModel
import uvicorn
import argparse
import time
import threading
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--netuid", type=str, default=1)
    parser.add_argument("--min_stake", type=int, default=100)
    parser.add_argument(
        "--chain_endpoint",
        type=str,
        default="subtensor_fixed_imagenet.thinkiftechnology.com:9944",
    )
    parser.add_argument("--disable_secure", action="store_true")
    args = parser.parse_args()
    return args


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed


class Data(BaseModel):
    prompt: str = "an image of"
    seed: int = 0
    max_length: int = 77
    additional_params: dict = {}


app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ARGS = get_args()
generator = pipeline(model="Gustavosta/MagicPrompt-Stable-Diffusion", device="cuda")


@app.middleware("http")
@limiter.limit("30/minute")
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


@app.post("/prompt_generate")
async def get_rewards(data: Data):
    seed_everything(data.seed)
    prompt = generator(
        data.prompt,
        max_length=data.max_length,
        **data.additional_params,
    )[0]["generated_text"]
    print("Prompt Generated:", prompt, flush=True)
    return {"prompt": prompt}


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
