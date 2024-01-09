from dependency_modules.rewarding.app import MODEL as pipe
import random
import torch
import os

import numpy as np

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

os.makedirs("tests/images/", exist_ok=True)
prompt = "an image of a japanese demon wearing a kimono, with demon horns and fire in both hands, dramatic lighting, illustration by Gr eg rutkowski, yoji shinkawa, 4k,"

for i in range(40):
    seed = i
    generator = torch.manual_seed(seed)
    result = pipe(prompt, generator=generator, num_inference_steps=30)
    images = result.images
    images[0].save(f"tests/images/{i}.webp")
