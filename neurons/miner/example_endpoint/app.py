from fastapi import FastAPI
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import torch
from typing import List
from utils import pil_image_to_base64
from pydantic import BaseModel
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--port', type=int, required=False, default=10006)

args = parser.parse_args()

class Prompt(BaseModel):
    prompt: str
    seed: int
    additional_params: dict = {}


app = FastAPI()
pipe = StableDiffusionPipeline.from_single_file("model.safetensors")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")


@app.get("/info")
async def get_model_name():
    return {"model_name": "RealisticVision"}


@app.post("/generate")
async def get_rewards(data: Prompt):
    generator = torch.Generator().manual_seed(data.seed)
    image = pipe(
        prompt=data.prompt, generator=generator, **data.additional_params
    ).images[0]
    image = pil_image_to_base64(image)
    return {"image": image}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port)
