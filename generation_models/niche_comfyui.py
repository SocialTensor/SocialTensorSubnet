from PIL import Image
from .base_model import BaseModel
import json
from generation_models import ComfyUI
import os
import shutil
from pathlib import Path
import random

def update_wf_sticker_maker(
    workflow,
    width=1024,
    height=1024,
    steps=20,
    prompt="a cute cat",
    negative_prompt="",
    seed=None,
    upscale_steps=10,
    is_upscale=True,
    **kwargs
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

class NicheComfyUI(BaseModel):
    def load_model(self, workflow_json_file, update_wf_func_name, **kwargs):
        comfyui = ComfyUI(random.randint(10000, 50000))
        output_folder = "generation_models/comfyui_helper/ComfyUI/output"
        input_folder = "generation_models/comfyui_helper/ComfyUI/input"
        comfyui_temp_output_folder = "generation_models/comfyui_helper/ComfyUI/temp"
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(input_folder, exist_ok=True)
        comfyui.start_server(output_folder, input_folder)
        workflow = json.load(open(workflow_json_file))
        comfyui.load_workflow(workflow)

        def inference_function(*args, **kwargs) -> Image.Image:
            self.cleanup(comfyui, output_folder, input_folder, comfyui_temp_output_folder)
            workflow = json.load(open(workflow_json_file))
            eval(update_wf_func_name)(workflow=workflow, **kwargs)
            wf = comfyui.load_workflow(workflow)
            comfyui.connect()
            comfyui.run_workflow(wf)
            files = []
            output_directories = [output_folder, comfyui_temp_output_folder]

            for directory in output_directories:
                print(f"Contents of {directory}:")
                files.extend(self.log_and_collect_files(directory))
            print(files)
            image = Image.open(files[0])
            return image
        
        return inference_function

    def __call__(self, *args, **kwargs):
        image: Image.Image = self.inference_function(*args, **kwargs)
        return image

    def cleanup(self, comfyui, output_folder, input_folder, comfyui_temp_output_folder):
        print("Cleanup")
        comfyui.clear_queue()
        for directory in [output_folder, input_folder, comfyui_temp_output_folder]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)


    def log_and_collect_files(self, directory, prefix=""):
        files = []
        for f in os.listdir(directory):
            if f == "__MACOSX":
                continue
            path = os.path.join(directory, f)
            if os.path.isfile(path):
                print(f"{prefix}{f}")
                files.append(Path(path))
            elif os.path.isdir(path):
                print(f"{prefix}{f}/")
                files.extend(self.log_and_collect_files(path, prefix=f"{prefix}{f}/"))
        return files

