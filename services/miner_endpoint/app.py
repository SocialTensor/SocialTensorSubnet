from fastapi import FastAPI
from typing import Union, Optional
from pydantic import BaseModel, Extra
import argparse
import uvicorn
from ray.serve.handle import DeploymentHandle
from ray import serve
import yaml
from services.rays.image_generating import ModelDeployment

MODEL_CONFIG = yaml.load(
    open("generation_models/configs/model_config.yaml"), yaml.FullLoader
)


class TextToImagePrompt(BaseModel, extra=Extra.allow):
    prompt: str
    seed: int
    pipeline_params: Optional[dict] = {}


class ImageToImagePrompt(BaseModel, extra=Extra.allow):
    prompt: str
    init_image: str
    seed: int
    pipeline_params: Optional[dict] = {}


class ControlNetPrompt(BaseModel, extra=Extra.allow):
    prompt: str
    controlnet_image: str
    seed: int
    pipeline_params: Optional[dict] = {}


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
    parser.add_argument("--num_gpus", type=int, default=1)
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


class MinerEndpoint:
    def __init__(self, model_handle: DeploymentHandle):
        self.model_handle = model_handle
        self.app = FastAPI()
        self.app.add_api_route("/generate", self.generate, methods=["POST"])
        self.app.add_api_route("/info", self.info, methods=["GET"])

    async def generate(
        self, prompt: Union[TextToImagePrompt, ImageToImagePrompt, ControlNetPrompt]
    ):
        prompt_data = prompt.dict()
        print(prompt_data)
        base_64_image = await self.model_handle.generate.remote(prompt_data=prompt_data)
        return base_64_image

    async def info(self):
        return {
            "model_name": args.model_name,
            "category": args.category,
        }


if __name__ == "__main__":
    model_deployment = serve.deployment(
        ModelDeployment,
        name="deployment",
        num_replicas=args.num_gpus,
        ray_actor_options={"num_gpus": args.num_gpus},
    )
    serve.run(
        model_deployment.bind(
            MODEL_CONFIG[args.category][args.model_name],
        ),
        name=f"deployment-{args.category}-{args.model_name}",
    )
    model_handle = serve.get_deployment_handle(
        "deployment", f"deployment-{args.category}-{args.model_name}"
    )
    app = MinerEndpoint(model_handle)
    uvicorn.run(
        app.app,
        host="0.0.0.0",
        port=args.port,
    )
