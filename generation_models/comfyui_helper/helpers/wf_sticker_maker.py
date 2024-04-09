import random


def setup(*args, **kwargs):
    pass


def update_workflow(
    workflow,
    width=1024,
    height=1024,
    steps=20,
    prompt="a cute cat",
    negative_prompt="",
    seed=None,
    upscale_steps=10,
    is_upscale=False,
    **kwargs,
):
    if not seed:
        seed = random.randint(0, 1e9)
    loader = workflow["2"]["inputs"]
    loader["empty_latent_width"] = width
    loader["empty_latent_height"] = height
    loader["positive"] = f"Sticker, {prompt}, svg, solid color background"
    loader["negative"] = f"nsfw, nude, {negative_prompt}, photo, photography"

    sampler = workflow["4"]["inputs"]
    sampler["seed"] = seed
    sampler["steps"] = steps

    upscaler = workflow["11"]["inputs"]
    if is_upscale:
        del workflow["5"]
        del workflow["10"]
        upscaler["steps"] = upscale_steps
        upscaler["seed"] = seed
    else:
        del workflow["16"]
        del workflow["17"]
        del workflow["18"]
        del upscaler["image"]
        del upscaler["model"]
        del upscaler["positive"]
        del upscaler["negative"]
        del upscaler["vae"]
