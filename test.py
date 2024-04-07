from generation_models import NicheStickerMaker
import time

model = NicheStickerMaker("/root/code/NicheImage/generation_models/comfyui_helper/configs/sticker_maker/workflow.json")

time.sleep(10)
for i in range(5):
    start = time.time()
    print(model(is_upscale=False))
    end = time.time()
    print(f"{i}: {end - start}")