from .base_model import BaseModel
from .utils import (
    download_checkpoint,
    base64_to_pil_image,
    resize_divisible,
    set_scheduler,
)
import diffusers
import torch
import os


class StableDiffusionXLImageToImage(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model(self, checkpoint_file, download_url, **kwargs):
        if not os.path.exists(checkpoint_file):
            download_checkpoint(download_url, checkpoint_file)

        pipe = diffusers.StableDiffusionXLImg2ImgPipeline.from_single_file(
            checkpoint_file,
            use_safetensors=True,
            torch_dtype=torch.float16,
            load_safety_checker=False,
        )
        scheduler_name = kwargs.get("scheduler", "euler_a")
        pipe.scheduler = set_scheduler(scheduler_name, pipe.scheduler.config)
        pipe.to("cuda")

        def inference_function(*args, **kwargs):
            # Prepare Init Image
            base64_init_image = kwargs.get("conditional_image", None)
            init_image = base64_to_pil_image(base64_init_image)
            init_image = resize_divisible(init_image, 768)
            kwargs.update({"image": init_image})
            # End Prepare Init Image

            image = pipe(*args, **kwargs).images[0]
            return image

        return inference_function
