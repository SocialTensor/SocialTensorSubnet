import diffusers
from PIL import Image
from .base_model import BaseModel
from .utils import (
    download_checkpoint,
    base64_to_pil_image,
    resize_for_condition_image,
    set_scheduler,
)
import os
import torch


class NicheStableDiffusionXL(BaseModel):
    def load_model(self, checkpoint_file, download_url, supporting_pipelines, **kwargs):
        if not os.path.exists(checkpoint_file):
            download_checkpoint(download_url, checkpoint_file)

        txt2img_pipe = diffusers.StableDiffusionXLPipeline.from_single_file(
            checkpoint_file,
            use_safetensors=True,
            torch_dtype=torch.float16,
            load_safety_checker=False,
        )
        txt2img_pipe.to("cuda")
        scheduler_name = kwargs.get("scheduler", "euler_a")
        txt2img_pipe.scheduler = set_scheduler(
            scheduler_name, txt2img_pipe.scheduler.config
        )

        txt2img_pipe.to("cuda")

        img2img_pipe = self.load_img2img(txt2img_pipe.components, supporting_pipelines)
        pipelines = {
            "txt2img": txt2img_pipe,
            "img2img": img2img_pipe,
        }

        def inference_function(*args, **kwargs) -> Image.Image:
            pipeline_type = kwargs["pipeline_type"]
            pipeline = pipelines[pipeline_type]
            if not pipeline:
                raise ValueError(f"Pipeline type {pipeline_type} is not supported")
            return pipeline(*args, **kwargs).images[0]

        return inference_function

    def load_img2img(self, components, supporting_pipelines) -> callable:
        if "img2img" not in supporting_pipelines:
            return None
        img2img_pipe = diffusers.StableDiffusionXLImg2ImgPipeline(**components)
        img2img_pipe.to("cuda")

        def inference_function(*args, **kwargs):
            conditional_image = self.process_conditional_image(**kwargs)
            width, height = conditional_image.size
            kwargs.update(
                {
                    "conditional_image": conditional_image,
                    "width": width,
                    "height": height,
                }
            )
            return img2img_pipe(*args, **kwargs)

        return inference_function

    def process_conditional_image(self, **kwargs) -> Image.Image:
        conditional_image = kwargs["conditional_image"]
        conditional_image = base64_to_pil_image(conditional_image)
        resolution = kwargs.get("resolution", 768)
        conditional_image = resize_for_condition_image(conditional_image, resolution)
        return conditional_image
