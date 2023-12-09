from fastapi import FastAPI
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
from PIL import Image
import torch
from typing import List
from utils import pil_image_to_base64, base64_to_pil_image
from pydantic import BaseModel


class Prompt(BaseModel):
    prompt: str
    seed: int
    additional_params: dict = {}


app = FastAPI()
pipe = StableDiffusionPipeline.from_single_file("model.safetensors")
pipe.enable_model_cpu_offload()
pipe.to("cuda")


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

    uvicorn.run(app, host="0.0.0.0", port=10001)
