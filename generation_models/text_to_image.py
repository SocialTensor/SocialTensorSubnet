from .base_model import BaseModel
from .utils import (
    download_checkpoint,
    set_scheduler,
)
import diffusers
import torch
import os


class StableDiffusionTextToImage(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model(self, checkpoint_file, download_url, **kwargs):
        if not os.path.exists(checkpoint_file):
            download_checkpoint(download_url, checkpoint_file)

        pipe = diffusers.StableDiffusionPipeline.from_single_file(
            checkpoint_file,
            use_safetensors=True,
            torch_dtype=torch.float16,
            load_safety_checker=False,
        )
        scheduler_name = kwargs.get("scheduler", "euler_a")
        pipe.scheduler = set_scheduler(scheduler_name, pipe.scheduler.config)
        pipe.to("cuda")

        def inference_function(*args, **kwargs):
            return pipe(*args, **kwargs).images[0]

        return inference_function


class StableDiffusionXLTextToImage(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model(self, checkpoint_file, download_url, **kwargs):
        if not os.path.exists(checkpoint_file):
            download_checkpoint(download_url, checkpoint_file)

        pipe = diffusers.StableDiffusionXLPipeline.from_single_file(
            checkpoint_file,
            use_safetensors=True,
            torch_dtype=torch.float16,
            load_safety_checker=False,
        )
        scheduler_name = kwargs.get("scheduler", "euler_a")
        pipe.scheduler = set_scheduler(scheduler_name, pipe.scheduler.config)
        pipe.to("cuda")

        def inference_function(*args, **kwargs):
            return pipe(*args, **kwargs).images[0]

        return inference_function
