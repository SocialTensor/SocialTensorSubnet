import diffusers
from PIL import Image
from .base_model import BaseModel
from .utils import (
    set_scheduler
)
import os
import torch
import inspect

class NicheStableDiffusionV3(BaseModel):
    def load_model(self, repo_id, supporting_pipelines, **kwargs):
        txt2img_pipe = diffusers.StableDiffusion3Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        txt2img_pipe.to("cuda")
        scheduler_name = kwargs.get("scheduler", "fm_euler")
        txt2img_pipe.scheduler = set_scheduler(
            scheduler_name, txt2img_pipe.scheduler.config
        )

        pipelines = {
            "txt2img": txt2img_pipe,
        }

        def inference_function(*args, **kwargs) -> Image.Image:
            pipeline_type = kwargs["pipeline_type"]
            pipeline = pipelines.get(pipeline_type)
            parameters = [param.name for param in inspect.signature(pipeline).parameters.values()]
            valid_kwargs = {key: kwargs[key] for key in kwargs if key in parameters}
            
            if not pipeline:
                raise ValueError(f"Pipeline type {pipeline_type} is not supported")
            return pipeline(*args, **valid_kwargs).images[0]

        return inference_function

