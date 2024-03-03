from PIL import Image, ImageDraw
from typing import List
import random
from io import BytesIO
import asyncio

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
async def notice_discord(images, webhook, content=""):
    n_image = len(images)
    notice_image = make_image_grid(images, (1, n_image), (256, 256))
    notice_io = BytesIO()
    notice_image.save(notice_io, format="JPEG")
    notice_io.seek(0)
    webhook.content = content
    webhook.add_file(file=notice_io, filename="notice.jpg")
    await webhook.execute()