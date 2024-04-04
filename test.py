from generation_models import NicheStableDiffusion


params = {
    "checkpoint_file": "checkpoints/RealisticVision.safetensors",
"download_url": "https://civitai.com/api/download/models/130072?type=Model&format=SafeTensor&size=pruned&fp=fp16",
"scheduler": "dpm++2m",
"supporting_pipelines": ['txt2img']
}
pipe = NicheStableDiffusion(
    **params
)

input_dict = {
    "pipeline_type": "txt2img",
    "prompt": "a cat",
    "num_inference_steps": 25,
}

image = pipe(**input_dict)
image.save("debug.webp")