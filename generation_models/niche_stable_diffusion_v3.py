import diffusers
from PIL import Image
from .base_model import BaseModel
from .utils import (
    set_scheduler
)
import os
import torch


class NicheStableDiffusionV3(BaseModel):
    def load_model(self, repo_id, supporting_pipelines, **kwargs):
        txt2img_pipe = diffusers.StableDiffusion3Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        txt2img_pipe.to("cuda")
        scheduler_name = kwargs.get("scheduler", "euler_a")
        txt2img_pipe.scheduler = set_scheduler(
            scheduler_name, txt2img_pipe.scheduler.config
        )

        pipelines = {
            "txt2img": txt2img_pipe,
        }

        def inference_function(*args, **kwargs) -> Image.Image:
            pipeline_type = kwargs["pipeline_type"]
            pipeline = pipelines.get(pipeline_type)
            kwargs.pop("seed", None)
            kwargs.pop("pipeline_type", None)
            kwargs.pop("pipeline_params", None)
            if not pipeline:
                raise ValueError(f"Pipeline type {pipeline_type} is not supported")
            return pipeline(*args, **kwargs).images[0]

        return inference_function

