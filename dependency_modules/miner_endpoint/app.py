from fastapi import FastAPI
import torch
from typing import List
from pydantic import BaseModel
import argparse
from dependency_modules.rewarding.utils import (
    instantiate_from_config,
    pil_image_to_base64,
)
import yaml

MODEL_CONFIG = yaml.load(open("configs/model_config.yaml"), yaml.FullLoader)


class Prompt(BaseModel):
    prompt: str
    seed: int
    additional_params: dict = {}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10006)
    parser.add_argument(
        "--model_name",
        type=str,
        choices=list(MODEL_CONFIG.keys()),
    )
    args = parser.parse_known_args()[0]
    return args


args = get_args()


app = FastAPI()
pipe = instantiate_from_config(MODEL_CONFIG[args.model_name])
generator = torch.Generator("cuda")


@app.get("/info")
async def get_model_name():
    return {"model_name": args.model_name}


@app.post("/generate")
async def get_rewards(data: Prompt):
    generator.manual_seed(data.seed)
    image = pipe(
        prompt=data.prompt, generator=generator, **data.additional_params
    ).images[0]
    image = pil_image_to_base64(image)
    return {"image": image}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port)
