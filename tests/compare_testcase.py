from dependency_modules.rewarding.app import MODEL as pipe
from dependency_modules.rewarding.hash_compare import matching_image
import torch
from PIL import Image

prompt = "a portrait of a man with a beard"

results = []
for i in range(20):
    seed = i
    generator = torch.Generator().manual_seed(seed)
    result = pipe(prompt, generator=generator, num_inference_steps=25)
    images = result.images
    ref_image = Image.open(f"tests/images/{i}.webp")
    reward = matching_image(ref_image, images[0])
    results.append(reward)

print(f"Accuracy: {sum(results)/len(results)}")
    
