from utils import base64_to_pil_image
import imagehash
from PIL import Image
from typing import List


def matching_image(miner_image: Image.Image, validator_image: Image.Image) -> bool:
    miner_hash = imagehash.average_hash(miner_image)
    validator_hash = imagehash.average_hash(validator_image)
    return miner_hash == validator_hash


def infer_hash(validator_image: str, batched_miner_images: List[str]):
    rewards = []
    for miner_image in batched_miner_images:
        miner_image = base64_to_pil_image(miner_image)
        if miner_image is None:
            reward = False
        else:
            reward = matching_image(miner_image, validator_image)
        rewards.append(reward)
    print(rewards, flush=True)
    return rewards
