from utils import matching_images, base64_to_pil_image, measure_time
import imagehash
from PIL import Image
from typing import List


def matching_images(
    miner_images: List[Image.Image], validator_images: List[Image.Image]
) -> bool:
    miner_hashes = [
        imagehash.average_hash(image, hash_size=4) for image in miner_images
    ]
    validator_hashes = [
        imagehash.average_hash(image, hash_size=4) for image in validator_images
    ]
    matching = [
        miner_hash == validator_hash
        for miner_hash, validator_hash in zip(miner_hashes, validator_hashes)
    ]
    return all(matching)


def infer_hash(validator_images, batched_miner_images):
    rewards = []
    for miner_images in batched_miner_images:
        try:
            miner_images = [base64_to_pil_image(image) for image in miner_images]
            reward = matching_images(miner_images, validator_images)
        except Exception as e:
            print(e, flush=True)
            reward = 0
        rewards.append(reward)
    return rewards
