from fastapi import FastAPI
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import torch
from typing import List
from utils import pil_image_to_base64
from pydantic import BaseModel


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
    images = pipe(
        prompt=data.prompt, generator=generator, **data.additional_params
    ).images
    images = [pil_image_to_base64(image) for image in images]
    return {"images": images}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=10006)
