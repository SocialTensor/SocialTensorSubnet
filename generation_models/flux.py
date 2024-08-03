import numpy as np
import random
import torch
import diffusers
from .base_model import BaseModel
from PIL import Image
from transformers import T5EncoderModel, BitsAndBytesConfig
from pydantic import BaseModel
import time
import gc

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048
MAX_INFERENCE_STEPS = 8


def flush():
    gc.collect()
    torch.cuda.empty_cache()

class FluxSchnell:
    class FluxInput:
        def __init__( 
            self,  
            prompt = "A board with the text 'It's NicheImage Time!'",
            generator = torch.Generator().manual_seed(random.randint(0, MAX_SEED)),
            width: int = 1024,
            height: int = 1024,
            guidance_scale: float = 0.0,
            num_inference_steps: int = 4,
            **kwargs
        ):
            self.prompt = prompt
            self.generator = generator
            self.width = width
            self.height = height
            self.guidance_scale = guidance_scale
            self.num_inference_steps = num_inference_steps
            self._check_inputs()

        def _check_inputs(self):
            self.width = min(self.width, MAX_IMAGE_SIZE)
            self.height = min(self.height, MAX_IMAGE_SIZE)
            self.num_inference_steps = min(self.num_inference_steps, MAX_INFERENCE_STEPS)

    def __init__(self, **kwargs):
        t5_encoder = T5EncoderModel.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", subfolder="text_encoder_2", torch_dtype=torch.bfloat16
        )
        self.text_encoder = diffusers.DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            text_encoder_2=t5_encoder,
            transformer=None,
            vae=None,
        )
        pipeline = diffusers.DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", 
            torch_dtype=torch.bfloat16,
            revision="refs/pr/1",
            text_encoder_2=None,
            text_encoder=None,
        )
        pipeline.enable_model_cpu_offload()

        self.pipeline = pipeline

    @torch.inference_mode()
    def __call__(self, *args, **kwargs):
        inputs = self.FluxInput(**kwargs)
        self.text_encoder.to("cuda")
        start = time.time()
        (
            prompt_embeds,
            pooled_prompt_embeds,
            _,
        ) = self.text_encoder.encode_prompt(prompt=inputs.prompt, prompt_2=None, max_sequence_length=256)
        self.text_encoder.to("cpu")
        flush()
        print(f"Prompt encoding time: {time.time() - start}")
        output = self.pipeline(
            prompt_embeds=prompt_embeds.bfloat16(),
            pooled_prompt_embeds=pooled_prompt_embeds.bfloat16(),
            generator=inputs.generator,
            width=inputs.width,
            height=inputs.height,
            guidance_scale=inputs.guidance_scale,
            num_inference_steps=inputs.num_inference_steps
        )
        image = output.images[0]
        return image