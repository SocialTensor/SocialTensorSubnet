from PIL import Image
from generation_models.utils import pil_image_to_base64
import diffusers
import argparse
import litserve as ls
import torch

class OpenModel(ls.LitAPI):
    def __init__(self, model_id, num_inference_steps=30, guidance_scale=7.0):
        self.pipeline = diffusers.DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

    def setup(self, device):
        self.pipeline.to(device)

    def decode_request(self, request):
        prompt = request.get("prompt")
        width = request["pipeline_params"].get("width")
        height = request["pipeline_params"].get("height")
        return prompt, width, height

    def predict(self, prompt, width, height):
        image = self.pipeline(
            prompt=prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            width=width,
            height=height,
        ).images[0]
        return image

    def encode_response(self, image):
        base64_image = pil_image_to_base64(image, format="PNG")
        return base64_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_gpus", default=1, type=int
    )
    parser.add_argument(
        "--port", default=10006, type=int
    )
    parser.add_argument(
        "--model_id", default=""
    )

    args = parser.parse_args()

    core = OpenModel(args.model_id)

    server = ls.LitServer(core, accelerator="auto", max_batch_size=1, devices=args.num_gpus)
    server.run(port=args.port)