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
