from .base_model import BaseModel
from .utils import (
    download_checkpoint,
    base64_to_pil_image,
    resize_divisible,
    set_scheduler,
)
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)
import torch
import os
from controlnet_aux.processor import (
    CannyDetector,
    MidasDetector,
    MLSDdetector,
)


class StableDiffusionControlNetTextToImage(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model(self, checkpoint_file, download_url, **kwargs):
        if not os.path.exists(checkpoint_file):
            download_checkpoint(download_url, checkpoint_file)
        controlnets = [
            ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16
            ),
            ControlNetModel.from_pretrained(
                "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16
            ),
            ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_mlsd", torch_dtype=torch.float16
            ),
        ]
        processors = [
            CannyDetector(),
            MidasDetector.from_pretrained("lllyasviel/Annotators"),
            MLSDdetector.from_pretrained("lllyasviel/Annotators"),
        ]

        pipe = StableDiffusionControlNetPipeline.from_single_file(
            checkpoint_file,
            torch_dtype=torch.float16,
            controlnet=controlnets,
            load_safety_checker=False,
        )
        scheduler_name = kwargs.get("scheduler", "euler_a")
        pipe.scheduler = set_scheduler(scheduler_name, pipe.scheduler.config)
        pipe.to("cuda")

        def inference_function(*args, **kwargs):
            # Prepare Init Image
            base64_controlnet_image = kwargs.get("conditional_image", None)
            controlnet_image = base64_to_pil_image(base64_controlnet_image)
            controlnet_image = resize_divisible(controlnet_image, 688)
            controlnet_images = [p(controlnet_image) for p in processors]
            kwargs.update({"image": controlnet_images})
            # End Prepare Init Image
            outputs = pipe(*args, **kwargs)
            return outputs.images[0]

        return inference_function
