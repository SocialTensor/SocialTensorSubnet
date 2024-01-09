from dependency_modules.rewarding.utils import base64_to_pil_image, pil_image_to_base64
from PIL import Image
from typing import List
import timm
import torchvision.transforms as T
import torch.nn.functional as F


def get_transform(model_name):
    data_config = timm.get_pretrained_cfg(model_name).to_dict()
    mean = data_config["mean"]
    std = data_config["std"]
    transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    return transform


MODEL = timm.create_model("vgg19", pretrained=True, num_classes=0)
MODEL.to("cuda")
MODEL.eval()
TRANSFORM = get_transform("vgg19")
THRESHOLD = 0.9


def get_similarity(image_1, image_2):
    image_1 = TRANSFORM(image_1).unsqueeze(0).to("cuda")
    image_2 = TRANSFORM(image_2).unsqueeze(0).to("cuda")
    prob = F.cosine_similarity(MODEL(image_1), MODEL(image_2))
    print(prob.item(), flush=True)
    return prob.item() < THRESHOLD


def infer_hash(validator_image: Image.Image, batched_miner_images: List[str]):
    rewards = []
    for miner_image in batched_miner_images:
        miner_image = base64_to_pil_image(miner_image)
        validator_image = base64_to_pil_image(pil_image_to_base64(validator_image))
        if miner_image is None:
            reward = False
        else:
            reward = get_similarity(miner_image, validator_image)
        rewards.append(reward)
    print(rewards, flush=True)
    return rewards
