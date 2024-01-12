from dependency_modules.rewarding.models import BaseT2IModel
from dependency_modules.rewarding.utils import download_checkpoint
import diffusers
import torch
import os
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import CLIPFeatureExtractor
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
    cosine_distance,
)


class NicheSafetyChecker(StableDiffusionSafetyChecker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transforms.Compose([transforms.PILToTensor()])
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


class StableDiffusion(BaseT2IModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model(self, checkpoint_file, download_url):
        if not os.path.exists(checkpoint_file):
            download_checkpoint(download_url, checkpoint_file)

        pipe = diffusers.StableDiffusionPipeline.from_single_file(
            checkpoint_file,
            use_safetensors=True,
            torch_dtype=torch.float16,
            load_safety_checker=True,
        )
        pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )
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
            torch_dtype=torch.float16,
            load_safety_checker=True,
        )
        pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
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
