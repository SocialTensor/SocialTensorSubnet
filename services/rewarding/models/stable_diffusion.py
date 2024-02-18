from services.rewarding.models import BaseModel
from services.rewarding.models.utils import set_scheduler
from services.rewarding.utils import download_checkpoint
import diffusers
import torch
import os
import torch.nn as nn
from PIL import Image
from transformers import CLIPFeatureExtractor
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
    cosine_distance,
)


def base64_to_pil_image(base64_image):
    from io import BytesIO
    import base64
    import PIL.Image
    import numpy as np

    image = base64.b64decode(base64_image)
    image = BytesIO(image)
    image = PIL.Image.open(image)
    image = np.array(image)
    image = PIL.Image.fromarray(image).convert("RGB")
    return image


def resize_divisible(image, max_size=1024, divisible=16):
    W, H = image.size
    if W > H:
        W, H = max_size, int(max_size * H / W)
    else:
        W, H = int(max_size * W / H), max_size
    W = W - W % divisible
    H = H - H % divisible
    image = image.resize((W, H))
    return image


class NicheSafetyChecker(StableDiffusionSafetyChecker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="feature_extractor"
        )

    @torch.no_grad()
    def forward(self, clip_input):
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        special_cos_dist = (
            cosine_distance(image_embeds, self.special_care_embeds)
            .cpu()
            .float()
            .numpy()
        )
        cos_dist = (
            cosine_distance(image_embeds, self.concept_embeds).cpu().float().numpy()
        )

        result = []
        batch_size = image_embeds.shape[0]
        for i in range(batch_size):
            result_img = {
                "special_scores": {},
                "special_care": [],
                "concept_scores": {},
                "bad_concepts": [],
            }

            # increase this value to create a stronger `nfsw` filter
            # at the cost of increasing the possibility of filtering benign images
            adjustment = 0.0

            for concept_idx in range(len(special_cos_dist[0])):
                concept_cos = special_cos_dist[i][concept_idx]
                concept_threshold = self.special_care_embeds_weights[concept_idx].item()
                result_img["special_scores"][concept_idx] = round(
                    concept_cos - concept_threshold + adjustment, 3
                )
                if result_img["special_scores"][concept_idx] > 0:
                    result_img["special_care"].append(
                        {concept_idx, result_img["special_scores"][concept_idx]}
                    )
                    adjustment = 0.01

            for concept_idx in range(len(cos_dist[0])):
                concept_cos = cos_dist[i][concept_idx]
                concept_threshold = self.concept_embeds_weights[concept_idx].item()
                result_img["concept_scores"][concept_idx] = round(
                    concept_cos - concept_threshold + adjustment, 3
                )
                if result_img["concept_scores"][concept_idx] > 0:
                    result_img["bad_concepts"].append(concept_idx)

            result.append(result_img)

        has_nsfw_concepts = [len(res["bad_concepts"]) > 0 for res in result]
        return has_nsfw_concepts

    def run_check(self, image):
        clip_input = self.feature_extractor(image, return_tensors="pt").to("cuda")
        has_nsfw_concept = self.forward(clip_input.pixel_values.to("cuda"))[0]
        return has_nsfw_concept


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
            load_safety_checker=True,
        )
        scheduler_name = kwargs.get("scheduler", "euler_a")
        pipe.scheduler = set_scheduler(scheduler_name, pipe.scheduler.config)
        pipe.to("cuda")

        def inference_function(*args, **kwargs):
            return pipe(*args, **kwargs)

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
            load_safety_checker=True,
        )
        scheduler_name = kwargs.get("scheduler", "euler_a")
        pipe.scheduler = set_scheduler(scheduler_name, pipe.scheduler.config)
        pipe.to("cuda")

        safety_checker = NicheSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to("cuda")

        def inference_function(*args, **kwargs):
            outputs = pipe(*args, **kwargs)
            images = outputs.images
            W, H = images[0].size
            for i, image in enumerate(images):
                if safety_checker.run_check(image):
                    print("NSFW image detected")
                    images[i] = Image.new("RGB", (W, H), "black")
            outputs.images = images
            return outputs

        return inference_function


class StableDiffusionImageToImage(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model(self, checkpoint_file, download_url, **kwargs):
        if not os.path.exists(checkpoint_file):
            download_checkpoint(download_url, checkpoint_file)

        pipe = diffusers.StableDiffusionImg2ImgPipeline.from_single_file(
            checkpoint_file,
            use_safetensors=True,
            torch_dtype=torch.float16,
            load_safety_checker=True,
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

            outputs = pipe(*args, **kwargs)
            return outputs

        return inference_function


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
            load_safety_checker=True,
        )
        scheduler_name = kwargs.get("scheduler", "euler_a")
        pipe.scheduler = set_scheduler(scheduler_name, pipe.scheduler.config)
        pipe.to("cuda")

        safety_checker = NicheSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to("cuda")

        def inference_function(*args, **kwargs):
            # Prepare Init Image
            base64_init_image = kwargs.get("conditional_image", None)
            init_image = base64_to_pil_image(base64_init_image)
            init_image = resize_divisible(init_image, 1024)
            kwargs.update({"image": init_image})
            # End Prepare Init Image
            outputs = pipe(*args, **kwargs)
            images = outputs.images
            W, H = images[0].size
            for i, image in enumerate(images):
                if safety_checker.run_check(image):
                    print("NSFW image detected")
                    images[i] = Image.new("RGB", (W, H), "black")
            outputs.images = images
            return outputs

        return inference_function


class StableDiffusionControlNetTextToImage(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model(self, checkpoint_file, download_url, **kwargs):
        if not os.path.exists(checkpoint_file):
            download_checkpoint(download_url, checkpoint_file)
        from controlnet_aux.processor import Processor

        processor = Processor("canny")
        controlnet = diffusers.ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny",
            torch_dtype=torch.float16,
        )
        pipe = diffusers.StableDiffusionControlNetPipeline.from_single_file(
            checkpoint_file,
            use_safetensors=True,
            torch_dtype=torch.float16,
            load_safety_checker=True,
            controlnet=controlnet,
        )
        scheduler_name = kwargs.get("scheduler", "euler_a")
        pipe.scheduler = set_scheduler(scheduler_name, pipe.scheduler.config)
        pipe.to("cuda")

        def inference_function(*args, **kwargs):
            # Prepare Init Image
            base64_controlnet_image = kwargs.get("conditional_image", None)
            controlnet_image = base64_to_pil_image(base64_controlnet_image)
            controlnet_image = resize_divisible(controlnet_image, 768)
            controlnet_image = processor(controlnet_image, to_pil=True)
            kwargs.update({"image": controlnet_image})
            # End Prepare Init Image

            outputs = pipe(*args, **kwargs)
            return outputs

        return inference_function
