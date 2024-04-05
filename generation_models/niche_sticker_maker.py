from .base_model import BaseModel
from generation_models.utils_api_comfyui import *
import uuid
from PIL import Image
import io
import base64


server_address = "82.67.70.191:40892"
client_id = "13c08530-8911-4e38-8489-7cded8eddd9d"

class NicheStickerMaker(BaseModel):
    def __init__(self, *args, **kwargs):
        self.server_address, self.client_id = kwargs.get("server_address"), kwargs.get("client_id")

        self.inference_function = self.load_model(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.inference_function(*args, **kwargs)


    def load_model(self, *args, **kwargs):
        imagine_inference_function = self.load_image(*args, **kwargs)
        return imagine_inference_function
    
    def load_image(self, *args, **kwargs):
        # workflow = load_workflow("generation_models/sticker_maker.json")

        def inference_function(*args, **kwargs):
            with open("generation_models/workflow-json/sticker_maker.json", "r") as file:
                workflow_json = file.read()
  

            workflow = json.loads(workflow_json)
            workflow["2"]["inputs"]["positive"] = kwargs["prompt"]
            workflow["4"]["inputs"]["seed"] = kwargs["seed"]


            ws = websocket.WebSocket()
            ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))

            images = get_images(ws, workflow)
            imgs = []
            for node_id in images:
                for image_data in images[node_id]:
                    image = Image.open(io.BytesIO(image_data))
                    imgs.append(image)

            return imgs[0]

        return inference_function

if __name__=="__main__":
    params = {
    "supporting_pipelines": ['txt2img']
    }
    pipe = NicheStickerMaker(
        **params
    )

    input_dict = {
        "pipeline_type": "txt2img",
        "prompt": "a cat",
    }

    image = pipe(**input_dict)
    image.save("debug.webp")

