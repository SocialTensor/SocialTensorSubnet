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

def flush():
    gc.collect()
    torch.cuda.empty_cache()

class Flux:
    class FluxInput:
        def __init__( 
            self,  
            prompt: str = "A board with the text 'It's NicheImage Time!'",
            generator: str = torch.Generator().manual_seed(random.randint(0, MAX_SEED)),
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
            

    def __init__(self, **kwargs):
        t5_encoder = T5EncoderModel.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", subfolder="text_encoder_2", revision="refs/pr/7", torch_dtype=torch.bfloat16
        )
        self.text_encoder = diffusers.DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            text_encoder_2=t5_encoder,
            transformer=None,
            vae=None,
            revision="refs/pr/7",
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
        image.save("fig.jpg")
        return image