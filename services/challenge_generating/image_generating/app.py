import diffusers
import torch
from fastapi import FastAPI, Request, Response
import bittensor as bt
from pydantic import BaseModel
import uvicorn
import argparse
import time
import threading
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from services.rewarding.utils import pil_image_to_base64


class TextToImagePrompt(BaseModel):
    prompt: str
    negative_prompt: str = "bad image, low quality, blurry"


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


ARGS = get_args()


app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

pipe = diffusers.StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
)
pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config
)
pipe.to("cuda")


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
async def generate(
    data: TextToImagePrompt,
):
    image = pipe(
        prompt=data.prompt,
        negative_prompt=data.negative_prompt,
        height=512,
        width=512,
        num_inference_steps=25,
    ).images[0]
    image = pil_image_to_base64(image)
    return {"image": image}


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
