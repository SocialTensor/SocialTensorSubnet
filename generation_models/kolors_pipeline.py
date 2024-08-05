import numpy as np
import random
import torch
import time
import gc
import transformers
import diffusers
import os
from diffusers import KolorsPipeline
from controlnet_aux.processor import Processor
from .utils import resize_image, base64_to_pil_image
from .kolors import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
from optimum.quanto import freeze, qfloat8, quantize

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
            prompt="A board with the text 'It's NicheImage Time!'",
            generator=torch.Generator().manual_seed(random.randint(0, MAX_SEED)),
            width: int = 1024,
            height: int = 1024,
            guidance_scale: float = 5.0,
            num_inference_steps: int = 28,
            conditional_image: str = "",
            negative_prompt="",
            control_guidance_end=0.9,
            strength=1.0,
            controlnet_conditioning_scale: list = [0.8],
            ip_adapter_scale=0.0,
            pipeline_type="txt2img",
            **kwargs,
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
            else:
                conditional_image = Image.new("RGB", (512, 512), "black")
            self.conditional_image = conditional_image
            self.negative_prompt = negative_prompt
            self.pipeline_type: str = pipeline_type
            self.controlnet_conditioning_scale = controlnet_conditioning_scale
            self.control_guidance_end = control_guidance_end
            self.strength = strength
            self.ip_adapter_scale = ip_adapter_scale
            self._check_inputs()

        def _check_inputs(self):
            self.width = min(self.width, MAX_IMAGE_SIZE)
            self.height = min(self.height, MAX_IMAGE_SIZE)
            self.num_inference_steps = min(
                self.num_inference_steps, MAX_INFERENCE_STEPS
            )

    @torch.no_grad()
    def __init__(self, **kwargs):
        pipeline = KolorsPipeline.from_pretrained(
            "Kwai-Kolors/Kolors-diffusers", torch_dtype=torch.float16, variant="fp16"
        )
        self.image_encoder = (
            transformers.CLIPVisionModelWithProjection.from_pretrained(
                "Kwai-Kolors/Kolors-IP-Adapter-Plus",
                subfolder="image_encoder",
                ignore_mismatched_sizes=True,
            )
            .to(dtype=torch.float16)
            .to("cuda")
        )
        self.clip_image_processor = transformers.CLIPImageProcessor(
            size=336, crop_size=336
        )
        quantize(pipeline.text_encoder, weights=qfloat8)
        freeze(pipeline.text_encoder)
        if hasattr(pipeline.unet, "encoder_hid_proj"):
            pipeline.unet.text_encoder_hid_proj = pipeline.unet.encoder_hid_proj
        pipeline.to("cuda")
        self.pipeline = pipeline
        if hasattr(self.pipeline.unet, "encoder_hid_proj"):
            self.pipeline.unet.text_encoder_hid_proj = (
                self.pipeline.unet.encoder_hid_proj
            )
        self.pipeline.load_ip_adapter(
            "Kwai-Kolors/Kolors-IP-Adapter-Plus",
            subfolder="",
            weight_name=["ip_adapter_plus_general.bin"],
            low_cpu_mem_usage=True,
        )
        self.pipeline.feature_extractor = self.clip_image_processor
        self.pipeline.image_encoder = self.image_encoder

        self.ip_adapter_fastload_params = {
            "unet.encoder_hid_proj_ip_adapter": self.pipeline.unet.encoder_hid_proj,
            "unet.encoder_hid_dim_type": self.pipeline.unet.encoder_hid_dim_type,
            "unet.encoder_hid_proj_original": self.pipeline.unet.text_encoder_hid_proj,
            "unet.config.encoder_hid_dim_type": self.pipeline.unet.config.encoder_hid_dim_type,
        }

        canny_controlnet = ControlNetModel.from_pretrained(
            "Kwai-Kolors/Kolors-ControlNet-Canny", torch_dtype=torch.float16
        )
        depth_controlnet = ControlNetModel.from_pretrained(
            "Kwai-Kolors/Kolors-ControlNet-Depth", torch_dtype=torch.float16
        )
        # canny_controlnet.enable_model_cpu_offload()
        # depth_controlnet.enable_model_cpu_offload()
        controlnet_pipeline = StableDiffusionXLControlNetImg2ImgPipeline(
            vae=pipeline.vae,
            controlnet=[canny_controlnet, depth_controlnet],
            text_encoder=pipeline.text_encoder,
            tokenizer=pipeline.tokenizer,
            unet=pipeline.unet,
            scheduler=pipeline.scheduler,
            image_encoder=self.image_encoder,
            feature_extractor=self.clip_image_processor,
            force_zeros_for_empty_prompt=False,
        ).to("cuda")
        self.controlnet_pipeline = controlnet_pipeline
        canny_processor = Processor("canny")
        depth_processor = Processor("depth_midas")
        depth_processor.processor.to("cuda")
        self.processors = [canny_processor, depth_processor]

    @torch.inference_mode()
    def __call__(self, *args, **kwargs):
        inputs = self.KolorsInput(**kwargs)
        self.pipeline.set_ip_adapter_scale(inputs.ip_adapter_scale)
        if inputs.pipeline_type == "controlnet":
            processed_images = [
                processor(inputs.conditional_image) for processor in self.processors
            ]
            image = self.controlnet_pipeline(
                prompt=inputs.prompt,
                negative_prompt=inputs.negative_prompt,
                generator=inputs.generator,
                width=inputs.width,
                height=inputs.height,
                guidance_scale=inputs.guidance_scale,
                num_inference_steps=inputs.num_inference_steps,
                image=inputs.conditional_image,
                control_image=processed_images,
                controlnet_conditioning_scale=inputs.controlnet_conditioning_scale,
                control_guidance_end=inputs.control_guidance_end,
                strength=inputs.strength,
                ip_adapter_image=inputs.conditional_image,
            ).images[0]

        elif inputs.pipeline_type in ["ip_adapter", "txt2img"]:
            start = time.time()
            image = self.pipeline(
                prompt=inputs.prompt,
                negative_prompt=inputs.negative_prompt,
                generator=inputs.generator,
                width=inputs.width,
                height=inputs.height,
                guidance_scale=inputs.guidance_scale,
                num_inference_steps=inputs.num_inference_steps,
                ip_adapter_image=inputs.conditional_image,
            ).images[0]
            self.pipeline.feature_extractor = None
            self.pipeline.image_encoder = None
            start = time.time()
            self.pipeline.unload_ip_adapter()
            print(f"Unloading IP Adapter took {time.time()-start:.2f} seconds")
        return image
