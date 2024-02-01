from fastapi import FastAPI
import torch
from typing import List, Union
from pydantic import BaseModel
import argparse
from services.rewarding.utils import (
    instantiate_from_config,
    pil_image_to_base64,
)
import yaml
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

MODEL_CONFIG = yaml.load(open("configs/model_config.yaml"), yaml.FullLoader)


class TextToImagePrompt(BaseModel):
    prompt: str
    seed: int
    pipeline_params: dict = {}


class ImageToImagePrompt(BaseModel):
    prompt: str
    init_image: str
    seed: int
    pipeline_params: dict = {}


class ControlNetPrompt(BaseModel):
    prompt: str
    controlnet_image: str
    seed: int
    pipeline_params: dict = {}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10006)
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


args = get_args()


app = FastAPI()
pipe = instantiate_from_config(MODEL_CONFIG[args.category][args.model_name])


@app.get("/info")
async def get_model_name():
    return {"model_name": args.model_name}


@app.post("/generate")
async def generate(
    data: Union[TextToImagePrompt, ImageToImagePrompt, ControlNetPrompt]
):
    generator = torch.manual_seed(data.seed)
    image = pipe(generator=generator, **data, **data.additional_params).images[0]
    image = pil_image_to_base64(image)
    return {"image": image}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port)
