import io
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import importlib
import requests
import os
from tqdm import tqdm


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def pil_image_to_base64(image: Image.Image) -> str:
    image_stream = io.BytesIO()
    image.save(image_stream, format="PNG")
    base64_image = base64.b64encode(image_stream.getvalue()).decode("utf-8")

    return base64_image


def base64_to_pil_image(base64_image: str) -> Image.Image:
    image_stream = io.BytesIO(base64.b64decode(base64_image))
    image = Image.open(image_stream)

    return image


def download_checkpoint(download_url, checkpoint_file):
    folder, filename = os.path.split(checkpoint_file)
    os.makedirs(folder, exist_ok=True)
    with requests.get(download_url, stream=True) as response:
        total_size = int(response.headers.get("content-length", 0))
        with open(checkpoint_file, "wb") as file_stream, tqdm(
            desc=checkpoint_file,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                file_stream.write(data)
                progress_bar.update(len(data))

    print("Download completed successfully.")
