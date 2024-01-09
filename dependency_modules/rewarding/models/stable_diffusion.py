from dependency_modules.rewarding.models import BaseT2IModel
from dependency_modules.rewarding.utils import download_checkpoint
import diffusers
import torch
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


class StableDiffusion(BaseT2IModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model(self, checkpoint_file, download_url):
        if not os.path.exists(checkpoint_file):
            download_checkpoint(download_url, checkpoint_file)

        pipe = diffusers.StableDiffusionPipeline.from_single_file(
            checkpoint_file,
            use_safetensors=True,
            load_safety_checker=False,
        )
        pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
        pipe.unet.set_default_attn_processor()
        pipe.vae.set_default_attn_processor()
        pipe.disable_xformers_memory_efficient_attention()
        pipe.to("cuda")

        def inference_function(*args, **kwargs):
            return pipe(*args, **kwargs)

        return inference_function


class StableDiffusionXL(BaseT2IModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model(self, checkpoint_file, download_url):
        if not os.path.exists(checkpoint_file):
            download_checkpoint(download_url, checkpoint_file)

        pipe = diffusers.StableDiffusionXLPipeline.from_single_file(
            checkpoint_file,
            use_safetensors=True,
            load_safety_checker=False,
        )
        pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
        pipe.unet.set_default_attn_processor()
        pipe.vae.set_default_attn_processor()
        pipe.disable_xformers_memory_efficient_attention()
        pipe.to("cuda")

        def inference_function(*args, **kwargs):
            return pipe(*args, **kwargs)

        return inference_function
