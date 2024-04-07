from PIL import Image
from .base_model import BaseModel
import json
from generation_models import ComfyUI
import os
from threading import Thread
import shutil
from pathlib import Path

class NicheStickerMaker(BaseModel):
    def load_model(self, workflow_json_file, **kwargs):

        comfyui = ComfyUI("127.0.0.1:8188")
        output_folder = "tmp/output"
        input_folder = "tmp/input"
        comfyui_temp_output_folder = "tmp/comfyui_temp_output"
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(input_folder, exist_ok=True)
        comfyui_node_thread = Thread(target=comfyui.start_server, args=(output_folder, input_folder))
        comfyui_node_thread.start()

        workflow = json.load(open(workflow_json_file))
        comfyui.load_workflow(workflow)

        def inference_function(*args, **kwargs) -> Image.Image:
            self.cleanup(comfyui, output_folder, input_folder, comfyui_temp_output_folder)
            self.update_workflow(workflow=workflow, **kwargs)
            wf = comfyui.load_workflow(workflow)
            comfyui.connect()
            comfyui.run_workflow(wf)
            files = []
            output_directories = [output_folder]

            for directory in output_directories:
                print(f"Contents of {directory}:")
                files.extend(self.log_and_collect_files(directory))

            image = Image.open(files[0])
            return image
        
        return inference_function

    def cleanup(self, comfyui, output_folder, input_folder, comfyui_temp_output_folder):
        comfyui.clear_queue()
        for directory in [output_folder, input_folder, comfyui_temp_output_folder]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

    def update_workflow(
        self,
        workflow,
        width=1024,
        height=1024,
        steps=20,
        prompt="a cute cat",
        negative_prompt="",
        seed=1,
        upscale_steps=10,
        is_upscale=False,
    ):
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

