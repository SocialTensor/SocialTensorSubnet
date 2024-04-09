import base64
import io
from PIL import Image
import random
import os
import shutil
from pathlib import Path
from PIL import ExifTags


LORA_WEIGHTS_MAPPING = {
    "3D": "artificialguybr/3DRedmond-3DRenderStyle-3DRenderAF.safetensors",
    "Emoji": "fofr/emoji.safetensors",
    "Video game": "artificialguybr/PS1Redmond-PS1Game-Playstation1Graphics.safetensors",
    "Pixels": "artificialguybr/PixelArtRedmond-Lite64.safetensors",
    "Clay": "artificialguybr/ClayAnimationRedm.safetensors",
    "Toy": "artificialguybr/ToyRedmond-FnkRedmAF.safetensors",
}

LORA_TYPES = list(LORA_WEIGHTS_MAPPING.keys())


def handle_input_file(input_file: Path, input_folder):
    file_extension = os.path.splitext(input_file)[1].lower()
    if file_extension in [".jpg", ".jpeg"]:
        filename = "input.png"
        image = Image.open(input_file)

        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == "Orientation":
                    break
            exif = dict(image._getexif().items())

            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
        except (KeyError, AttributeError):
            # EXIF data does not have orientation
            # Do not rotate
            pass

        image.save(os.path.join(input_folder, filename))
    elif file_extension in [".png", ".webp"]:
        filename = f"input{file_extension}"
        shutil.copy(input_file, os.path.join(input_folder, filename))
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    return filename


def base64_to_image(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))


def download_loras(comfyui, **kwargs):
    for weight in LORA_WEIGHTS_MAPPING.values():
        comfyui.weights_downloader.download_weights(weight)


def setup(*args, **kwargs):
    download_loras(*args, **kwargs)


def update_workflow(
    workflow: dict,
    conditional_image: str,
    input_folder,
    prompt="a person",
    style="3D",
    negative_prompt="",
    control_depth_strength=0.8,
    lora_scale=1,
    instant_id_strength=1,
    denoising_strength=0.65,
    seed=None,
    prompt_strength=4.5,
    **kwargs,
):
    if not seed:
        seed = random.randint(0, 1e9)
    image = base64_to_image(conditional_image)
    image_file = f"{input_folder}/image.jpg"
    image.save(image_file)
    filename = handle_input_file(image_file, input_folder)
    lora_name = LORA_WEIGHTS_MAPPING[style]
    prompt = style_to_prompt(style, prompt)
    negative_prompt = style_to_negative_prompt(style, negative_prompt)
    load_image = workflow["22"]["inputs"]
    load_image["image"] = filename

    loader = workflow["2"]["inputs"]
    loader["positive"] = prompt
    loader["negative"] = negative_prompt

    controlnet = workflow["28"]["inputs"]
    controlnet["strength"] = control_depth_strength

    lora_loader = workflow["3"]["inputs"]
    lora_loader["lora_name_1"] = lora_name
    lora_loader["lora_wt_1"] = lora_scale

    instant_id = workflow["41"]["inputs"]
    instant_id["weight"] = instant_id_strength

    sampler = workflow["4"]["inputs"]
    sampler["denoise"] = denoising_strength
    sampler["seed"] = seed
    sampler["cfg"] = prompt_strength


def style_to_prompt(style, prompt):
    style_prompts = {
        "3D": f"3D Render Style, 3DRenderAF, {prompt}",
        "Emoji": f"memoji, emoji, {prompt}, 3d render, sharp",
        "Video game": f"Playstation 1 Graphics, PS1 Game, {prompt}, Video game screenshot",
        "Pixels": f"Pixel Art, PixArFK, {prompt}",
        "Clay": f"Clay Animation, Clay, {prompt}",
        "Toy": f"FnkRedmAF, {prompt}, toy, miniature",
    }
    return style_prompts[style]


def style_to_negative_prompt(style, negative_prompt=""):
    if negative_prompt:
        negative_prompt = f"{negative_prompt}, "

    start_base_negative = "nsfw, nude, oversaturated, "
    end_base_negative = "ugly, broken, watermark"
    specifics = {
        "3D": "photo, photography, ",
        "Emoji": "photo, photography, blurry, soft, ",
        "Video game": "text, photo, ",
        "Pixels": "photo, photography, ",
        "Clay": "",
        "Toy": "",
    }

    return (
        f"{specifics[style]}{start_base_negative}{negative_prompt}{end_base_negative}"
    )
