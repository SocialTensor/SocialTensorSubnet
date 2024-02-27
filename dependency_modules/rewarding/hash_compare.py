from dependency_modules.rewarding.utils import base64_to_pil_image, pil_image_to_base64
import imagehash
from PIL import Image
from typing import List


def get_black_hash(H, W) -> str:
    image = Image.new("RGB", (W, H), color="black")
    return str(imagehash.average_hash(image, hash_size=6))


def matching_image(miner_image: Image.Image, validator_image: Image.Image) -> bool:
    miner_hash = imagehash.average_hash(miner_image, hash_size=6)
    validator_hash = imagehash.average_hash(validator_image, hash_size=6)
    print("Hamming Distance:", miner_hash - validator_hash, flush=True)
    return (miner_hash - validator_hash) <= 3


def nsfw_filter(validator_image: Image.Image, miner_image: Image.Image) -> bool:
    W, H = validator_image.size
    validator_hash = imagehash.average_hash(validator_image, hash_size=6)
    miner_hash = imagehash.average_hash(miner_image, hash_size=6)
    validator_hash = str(validator_hash)
    miner_hash = str(miner_hash)
    black_hash = get_black_hash(H, W)
    if validator_hash != black_hash and miner_hash == black_hash:
        return 1
    if validator_hash == black_hash and miner_hash != black_hash:
        return 2
    return 0


def infer_hash(validator_image: Image.Image, batched_miner_images: List[str]):
    rewards = []
    for miner_image in batched_miner_images:
        miner_image = base64_to_pil_image(miner_image)
        validator_image = base64_to_pil_image(pil_image_to_base64(validator_image))
        if miner_image is None:
            reward = False
        else:
            nsfw_check = nsfw_filter(validator_image, miner_image)
            if nsfw_check:
                reward = -5 if nsfw_check == 2 else 0
            else:
                reward = matching_image(miner_image, validator_image)
        rewards.append(reward)
    print(rewards, flush=True)
    return rewards
