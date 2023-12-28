import io
import numpy as np
import base64
from PIL import Image
from io import BytesIO


def pil_image_to_base64(image: Image.Image) -> str:
    image_stream = io.BytesIO()    
    image.save(image_stream, format="PNG")    
    base64_image = base64.b64encode(image_stream.getvalue()).decode('utf-8')
    
    return base64_image

def base64_to_pil_image(base64_image: str) -> Image.Image:
    image_stream = io.BytesIO(base64.b64decode(base64_image))
    image = Image.open(image_stream)
    
    return image