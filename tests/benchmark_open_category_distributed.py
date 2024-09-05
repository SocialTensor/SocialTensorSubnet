import image_generation_subnet as ig
import requests
from generation_models.utils import base64_to_pil_image

synapse = ig.protocol.ImageGenerating(model_name="OpenCategory", timeout=12)

sample_prompts = [
    "a cat is sitting on a chair, behind is a window with a view of a garden.",
    'a landscape photo of a large, open field with a clear blue sky. The field appears to be a mix of grass and small rocks, with some areas showing signs of dryness and patches of vegetation. The sky is partly cloudy, with a few clouds scattered across it.   In the foreground, there is a small, open field with a few scattered trees and shrubs. The ground is covered with a mix of grass and small rocks, and there are some trees and shrubs in the distance. The sky is overcast, with a grayish hue, suggesting a cloudy day. The overall scene suggests a rural or agricultural setting.  In the bottom left corner of the image, there is a text that reads "ENHANCED BY BABELCOLOUR," indicating that the image has been digitally enhanced or enhanced by a specific software or technique.  The text is in black and white, which adds a timeless quality to the image.  The overall composition of the image is simple yet striking, focusing on the field and the people in the foreground.',
    "a photo of a young man with a serious expression. He has short, dark hair and is wearing a light-colored, long-sleeved shirt with a collar. His left hand is raised to his face, with his index finger resting on his cheek and his thumb extended. He is wearing dark sunglasses and a light-colored, long-sleeved shirt. The background is blurred but appears to be an outdoor setting with trees and a clear sky. The photograph has a vintage feel, likely from the mid-20th century.   1. The young man has a beard and mustache, and his hair is neatly combed. He is wearing a dark-colored, long-sleeved shirt with a collar. His left hand is raised to his face, with his index finger resting on his cheek and his thumb extended. He is wearing dark sunglasses and a light-colored, long-sleeved shirt. The background is slightly blurred but shows a natural setting with trees and a clear sky.",
]

from services.rewarding.open_category_reward import OpenCategoryReward

rewarder = OpenCategoryReward()

for prompt in sample_prompts:
    print("Prompt: ", prompt)
    synapse.prompt = prompt
    synapse.pipeline_params = {
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 4,
    }
    response = requests.post(
        "http://localhost:10006/generate", json=synapse.deserialize_input()
    )
    response.raise_for_status()
    response = response.json()
    image = base64_to_pil_image(response["image"])
    image.save("tmp.png")
    print("Image saved to tmp.png")
    reward = rewarder.get_reward(prompt, [response["image"]])
    print(reward)
