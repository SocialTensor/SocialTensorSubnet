from dependency_modules.rewarding.app import MODEL as pipe
from dependency_modules.rewarding.hash_compare import matching_image
import torch
import os
from PIL import Image

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

prompt = "a portrait of a man with a beard"
os.makedirs("tests/error/", exist_ok=True)


results = []
for i in range(40):
    print(i)
    seed = i
    generator = torch.Generator().manual_seed(seed)
    result = pipe(prompt, generator=generator, num_inference_steps=40)
    images = result.images
    images[0].save("tests/image.webp")
    image = Image.open("tests/image.webp")
    ref_image = Image.open(f"tests/images/{i}.webp")
    reward = matching_image(ref_image, image)
    if not reward:
        print(f"Error: {i}")
        ref_image.save(f"tests/error/{i}_ref.webp")
        image.save(f"tests/error/{i}_image.webp")
    results.append(reward)

print(f"Accuracy: {sum(results)/len(results)}")
