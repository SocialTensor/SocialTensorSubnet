import numpy as np
import random
import torch
import diffusers
from PIL import Image
from pydantic import BaseModel
import time
import gc
import os
from diffusers import KolorsPipeline
from controlnet_aux.processor import Processor
from .utils import resize_image, base64_to_pil_image
from .kolors import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048
MAX_INFERENCE_STEPS = 50

def flush():
    gc.collect()
    torch.cuda.empty_cache()

class Kolors:
    class KolorsInput:
        def __init__( 
            self,  
            prompt = "A board with the text 'It's NicheImage Time!'",
            generator = torch.Generator().manual_seed(random.randint(0, MAX_SEED)),
            width: int = 1024,
            height: int = 1024,
            guidance_scale: float = 6.0,
            num_inference_steps: int = 28,
            conditional_image: str = "",
            negative_prompt = "nsfw，脸部阴影，低分辨率，jpeg伪影、模糊、糟糕，黑脸，霓虹灯",
            controlnet_conditioning_scale = 0.7,
            control_guidance_end = 0.9,
            strength = 1.0,
            controlnet_conditioning_scale: list = [0.5, 0.5],
            pipeline_type="txt2img",
            **kwargs
        ):
            self.prompt = prompt
            self.generator = generator
            self.width = width
            self.height = height
            self.guidance_scale = guidance_scale
            self.num_inference_steps = num_inference_steps
            if conditional_image:
                conditional_image = base64_to_pil_image(conditional_image)
                conditional_image = resize_image(conditional_image, MAX_IMAGE_SIZE)
            self.conditional_image = conditional_image
            self.negative_prompt = negative_prompt
            self.pipeline_type: str = pipeline_type
            self.controlnet_conditioning_scale = controlnet_conditioning_scale
            self._check_inputs()

        def _check_inputs(self):
            self.width = min(self.width, MAX_IMAGE_SIZE)
            self.height = min(self.height, MAX_IMAGE_SIZE)
            self.num_inference_steps = min(self.num_inference_steps, MAX_INFERENCE_STEPS)

    def __init__(self, **kwargs):
        pipeline = KolorsPipeline.from_pretrained(
            "Kwai-Kolors/Kolors-diffusers", 
            torch_dtype=torch.float16, 
            variant="fp16"
        ).to("cuda")
        self.pipeline = pipeline

        canny_controlnet = ControlNetModel.from_pretrained("Kwai-Kolors/Kolors-ControlNet-Canny", torch_dtype=torch.float16)
        depth_controlnet = ControlNetModel.from_pretrained("Kwai-Kolors/Kolors-ControlNet-Depth", torch_dtype=torch.float16)
        controlnet_pipeline = StableDiffusionXLControlNetImg2ImgPipeline(
            vae=pipeline.vae,
            controlnet = [canny_controlnet, depth_controlnet],
            text_encoder=pipeline.text_encoder,
            tokenizer=pipeline.tokenizer,
            unet=pipeline.unet,
            scheduler=pipeline.scheduler,
            force_zeros_for_empty_prompt=False
        )
        controlnet_pipeline = controlnet_pipeline.to("cuda")
        self.controlnet_pipeline = controlnet_pipeline
        canny_processor = Processor("canny")
        depth_processor = Processor("depth_midas")
        depth_processor.to("cuda")
        self.processors = [canny_processor, depth_processor]


    @torch.inference_mode()
    def __call__(self, *args, **kwargs):
        inputs = self.FluxInput(**kwargs)
        if inputs.pipeline_type == "txt2img":
            image = self.pipeline(
                prompt=inputs.prompt,
                negative_prompt=inputs.negative_prompt,
                generator=inputs.generator,
                width=inputs.width,
                height=inputs.height,
                guidance_scale=inputs.guidance_scale,
                num_inference_steps=inputs.num_inference_steps,
            ).images[0]
        elif inputs.pipeline == "controlnet":
            processed_images = [processor(inputs.conditional_image) for processor in self.processors]
            image = pipe(
                prompt=inputs.prompt,
                negative_prompt=inputs.negative_prompt,
                generator=inputs.generator,
                width=inputs.width,
                height=inputs.height,
                guidance_scale=inputs.guidance_scale,
                num_inference_steps=inputs.num_inference_steps,
                image =inputs.conditional_image,
                control_image=processed_images,
                controlnet_conditioning_scale=inputs.controlnet_conditioning_scale,
                control_guidance_end = inputs.control_guidance_end, 
                strength=inputs.strength, 
                negative_prompt=inputs.negative_prompt , 
                num_inference_steps=inputs.num_inference_steps, 
                guidance_scale=inputs.guidance_scale,
            ).images[0]
        return image