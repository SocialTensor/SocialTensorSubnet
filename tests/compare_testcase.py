from dependency_modules.rewarding.app import MODEL as pipe
from dependency_modules.rewarding.similarity_compare import get_similarity
import torch
import os
from PIL import Image
import shutil

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

prompt = "an image of a japanese demon wearing a kimono, with demon horns and fire in both hands, dramatic lighting, illustration by Gr eg rutkowski, yoji shinkawa, 4k,"
shutil.rmtree("tests/error/")
os.makedirs("tests/error/", exist_ok=True)


results = []
for i in range(40):
    print(i)
    torch.manual_seed(0)
    seed = i
    generator = torch.manual_seed(seed)
    result = pipe(prompt, generator=generator, num_inference_steps=30)
    images = result.images
    images[0].save("tests/image.webp")
    image = Image.open("tests/image.webp")
    ref_image = Image.open(f"tests/images/{i}.webp")
    reward = get_similarity(image, ref_image)
    if not reward:
        print(f"Error: {i}")
        ref_image.save(f"tests/error/{i}_ref.png")
        image.save(f"tests/error/{i}_image.png")
    results.append(reward)
results = torch.tensor(results)
print(f"Mean: {results.mean()}")
print(f"Std: {results.std()}")
