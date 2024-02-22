from diffusers import (
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    LCMScheduler,
)
from io import BytesIO
import base64
import PIL.Image
from PIL import Image
import numpy as np
import requests
import os
from tqdm import tqdm
import io
import importlib


def instantiate_from_config(config):
    if "target" not in config:
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


def set_scheduler(scheduler_name: str, config):
    if scheduler_name == "euler":
        scheduler = EulerDiscreteScheduler.from_config(config)
    elif scheduler_name == "euler_a":
        scheduler = EulerAncestralDiscreteScheduler.from_config(config)
    elif scheduler_name == "dpm++2m_karras":
        scheduler = DPMSolverMultistepScheduler.from_config(
            config, use_karras_sigmas=True
        )
    elif scheduler_name == "dpm++sde_karras":
        scheduler = DPMSolverMultistepScheduler.from_config(
            config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
        )
    elif scheduler_name == "dpm++2m":
        scheduler = DPMSolverMultistepScheduler.from_config(config)
    elif scheduler_name == "dpm++sde":
        scheduler = DPMSolverMultistepScheduler.from_config(
            config, algorithm_type="sde-dpmsolver++"
        )
    elif scheduler_name == "lcm":
        scheduler = LCMScheduler.from_config(config)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return scheduler


def base64_to_pil_image(base64_image):
    image = base64.b64decode(base64_image)
    image = BytesIO(image)
    image = PIL.Image.open(image)
    image = np.array(image)
    image = PIL.Image.fromarray(image).convert("RGB")
    return image


def pil_image_to_base64(image: Image.Image) -> str:
    image_stream = io.BytesIO()
    image.save(image_stream, format="PNG")
    base64_image = base64.b64encode(image_stream.getvalue()).decode("utf-8")

    return base64_image


def resize_divisible(image, max_size=1024, divisible=16):
    W, H = image.size
    if W > H:
        W, H = max_size, int(max_size * H / W)
    else:
        W, H = int(max_size * W / H), max_size
    W = W - W % divisible
    H = H - H % divisible
    image = image.resize((W, H))
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
