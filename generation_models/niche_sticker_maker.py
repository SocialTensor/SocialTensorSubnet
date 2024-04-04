from base_model import BaseModel
from generation_models.utils_api_comfyui import *

class NicheStickerMaker(BaseModel):
    def __init__(self, *args, **kwargs):
        self.server_address = "82.67.70.191:40830"
        self.client_id = str(uuid.uuid4())
        self.inference_function = self.load_model(*args, **kwargs)


    def load_model(self, *args, **kwargs):
        imagine_inference_function = self.load_image(*args, **kwargs)
        return imagine_inference_function
    
    def load_image(self, *args, **kwargs):
        workflow = load_workflow("sticker_maker.json")

        def inference_function(*args, **kwargs):
            workflow["2"]["inputs"]["positive"] = kwargs["prompt"]
            workflow["4"]["inputs"]["seed"] = seed


            ws = websocket.WebSocket()
            ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))



            images = get_images(ws, prompt)
            return images

        return inference_function

if __name__=="__main__":
    test_sticker = NicheStickerMaker({"prompt":"cute dog", "seed":7})
    test_sticker.load_model()

