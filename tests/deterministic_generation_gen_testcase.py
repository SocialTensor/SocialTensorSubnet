from dependency_modules.rewarding.app import MODEL as pipe
import random
import torch
import os

os.makedirs("tests/images/", exist_ok=True)
prompt = "a portrait of a man with a beard"

for i in range(20):
    seed = i
    generator = torch.Generator().manual_seed(seed)
    result = pipe(prompt, generator=generator, num_inference_steps=25)
    images = result.images
    images[0].save(f"tests/images/{i}.webp")
