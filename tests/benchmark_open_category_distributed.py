import image_generation_subnet as ig
import requests
from generation_models.utils import base64_to_pil_image

synapse = ig.protocol.ImageGenerating(model_name="OpenCategory", timeout=12)

sample_prompts = [
    "a cat is sitting on a chair, behind is a window with a view of a garden.",
    "a landscape photo of a large, open field with a clear blue sky. The field appears to be a mix of grass and small rocks, with some areas showing signs of dryness and patches of vegetation.",
    "a photo of a young man with a serious expression. He has short, dark hair and is wearing a light-colored, long-sleeved shirt with a collar. ",
]

from services.rewarding.open_category_reward import OpenCategoryReward

rewarder = OpenCategoryReward()

for i, prompt in enumerate(sample_prompts):
    print("Prompt: ", prompt)
    synapse.prompt = prompt
    synapse.pipeline_params = {
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 30,
    }
    response = requests.post(
        "http://localhost:10006/generate", json=synapse.deserialize_input()
    )
    response.raise_for_status()
    response = response.json()
    image = base64_to_pil_image(response["image"])
    image.save(f"tmp_{i}.png")
    print("Image saved to tmp.png")
    reward = rewarder.get_reward(prompt, [response["image"]])
    print(reward)
