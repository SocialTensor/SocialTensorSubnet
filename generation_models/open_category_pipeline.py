from PIL import Image
from generation_models.utils import pil_image_to_base64
import diffusers
import argparse
import litserve as ls
import torch

class OpenModel(ls.LitAPI):
    def __init__(self, category, model_id, num_inference_steps=30, guidance_scale=7.0):
        self.pipeline = diffusers.DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.model_id = model_id
        self.category = category
    
    def get_info(self):
        return {
            "model_name": self.category,
        }

    def setup(self, device):
        self.pipeline.to(device)

    def decode_request(self, request):
        prompt = request.get("prompt")
        width = request.get("pipeline_params", {}).get("width", 1024)
        height = request.get("pipeline_params", {}).get("height", 1024)
        params = {
            "prompt": prompt,
            "width": width,
            "height": height,
        }
        print(params)
        return params

    def predict(self, params):
        image = self.pipeline(
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            **params,
        ).images[0]
        return image

    def encode_response(self, image):
        base64_image = pil_image_to_base64(image, format="PNG")
        return base64_image