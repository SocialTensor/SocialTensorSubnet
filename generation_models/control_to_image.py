from .base_model import BaseModel
from .utils import (
    download_checkpoint,
    base64_to_pil_image,
    resize_divisible,
    set_scheduler,
)
from diffusers import (
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
    AutoencoderKL,
)
import torch
import os


class StableDiffusionXLControlNetTextToImage(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model(self, checkpoint_file, download_url, **kwargs):
        if not os.path.exists(checkpoint_file):
            download_checkpoint(download_url, checkpoint_file)
        from controlnet_aux.lineart import LineartDetector

        line_detector = LineartDetector.from_pretrained("lllyasviel/Annotators").to(
            "cuda"
        )
        adapter = T2IAdapter.from_pretrained(
            "TencentARC/t2i-adapter-lineart-sdxl-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        )
        pipe = StableDiffusionXLAdapterPipeline.from_single_file(
            checkpoint_file,
            vae=vae,
            adapter=adapter,
            torch_dtype=torch.float16,
            load_safety_checker=False,
        )
        scheduler_name = kwargs.get("scheduler", "euler_a")
        pipe.scheduler = set_scheduler(scheduler_name, pipe.scheduler.config)
        pipe.to("cuda")

        def inference_function(*args, **kwargs):
            # Prepare Init Image
            base64_controlnet_image = kwargs.get("conditional_image", None)
            controlnet_image = base64_to_pil_image(base64_controlnet_image)
            controlnet_image = resize_divisible(controlnet_image, 768)
            controlnet_image = line_detector(
                controlnet_image, detect_resolution=384, image_resolution=1024
            )
            kwargs.update({"image": controlnet_image})
            # End Prepare Init Image

            outputs = pipe(*args, **kwargs)
            return outputs.images[0]

        return inference_function
