from generation_models.utils import base64_to_pil_image
import imagehash
from PIL import Image, ImageDraw
from typing import List
import random
from io import BytesIO

def make_image_grid(image_list, grid_size, image_size):
    """
    Creates a grid of images and returns a single combined image.

    :param image_list: List of PIL Image objects.
    :param grid_size: Tuple of (rows, cols) for the grid.
    :param image_size: Tuple of (width, height) for each image.
    :return: PIL Image object representing the grid.
    """
    rows, cols = grid_size
    width, height = image_size
    grid_img = Image.new('RGB', (cols * width, rows * height))
    
    for index, img in enumerate(image_list):
        # Calculate the position of the current image in the grid
        row = index // cols
        col = index % cols
        box = (col * width, row * height, (col + 1) * width, (row + 1) * height)
        
        # Resize image if it does not match the specified size
        if img.size != image_size:
            img = img.resize(image_size)
        
        # Paste the current image into the grid
        grid_img.paste(img, box)
        
    return grid_img

def get_black_hash(H, W) -> str:
    image = Image.new("RGB", (W, H), color="black")
    return str(imagehash.average_hash(image, hash_size=8))


def matching_image(miner_image: Image.Image, validator_image: Image.Image) -> bool:
    miner_hash = imagehash.average_hash(miner_image, hash_size=8)
    validator_hash = imagehash.average_hash(validator_image, hash_size=8)
    print("Hamming Distance:", miner_hash - validator_hash, flush=True)
    return (miner_hash - validator_hash) <= 6


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


def infer_hash(validator_image: Image.Image, batched_miner_images: List[str], webhook = None, probability = 0.2):
    rewards = []
    validator_image = base64_to_pil_image(validator_image)
    for miner_image in batched_miner_images:
        if not miner_image:
            reward = False
        else:
            try:
                miner_image = base64_to_pil_image(miner_image)
            except Exception:
                print(f"Corrupted miner image", flush=True)
                reward = False
                rewards.append(reward)
                continue
            nsfw_check = nsfw_filter(validator_image, miner_image)
            if nsfw_check:
                reward = -5 if nsfw_check == 2 else 0
            else:
                reward = matching_image(miner_image, validator_image)
            if reward <= 0 and webhook and random.random() < probability:
                notice_image = make_image_grid([validator_image, miner_image], (1, 2), (256, 256))
                notice_io = BytesIO()
                notice_image.save(notice_io, format="JPEG")
                notice_io.seek(0)
                webhook.add_file(file=notice_io, filename="notice.jpg")
                webhook.execute()
        rewards.append(reward)
    print(rewards, flush=True)
    return rewards
