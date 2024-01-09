import requests
import json
import torch
from dependency_modules.rewarding.app import instantiate_from_config, MODEL_CONFIG
from image_generation_subnet.validator.forward import get_reward
from image_generation_subnet.protocol import ImageGenerating
from dependency_modules.rewarding.utils import pil_image_to_base64
import numpy as np
N = 10
url_dict = {
    "RealisticVision": "http://check_realistic_vision_nicheimage.nichetensor.com:15011/verify",
    "SDXLTurbo": "http://sdxl_turbo_nicheimage.nichetensor.com:15012/verify",
}
rewards = {}
for model, url in url_dict.items():
    pipe = instantiate_from_config(MODEL_CONFIG[model])
    rewards[model] = []
    if model == "RealisticVision":
        steps = 30
    else:
        steps = 4
    for i in range(N):
        prompt = "an image of samurai, red hat, and a katana, with a red background"
        seed = i
        synapse = ImageGenerating(
            prompt=prompt,
            seed=seed,
            model_name=model,
            additional_params={"num_inference_steps": steps},
        )
        generator = torch.manual_seed(seed)
        result = pipe(prompt, generator=generator, num_inference_steps=steps)
        images = result.images
        synapse.image = pil_image_to_base64(images[0])
        reward = get_reward(url, [synapse], synapse)[0]
        rewards[model].append(reward)

    print(f"Model: {model}, rewards: {rewards[model]}")
    print(f"Accuracy: {np.mean(rewards[model])}")


