from abc import ABC, abstractmethod
from transformers import pipeline
from PIL import Image
import torch
import os

class BaseModel(ABC):
    def __init__(self, *args, **kwargs):
        self.inference_function = self.load_model(*args, **kwargs)
        self.nsfw_classifier = pipeline(
            "image-classification",
            model="Falconsai/nsfw_image_detection",
            device="cuda",
        )
        self.nsfw_threshold = 0.5

    @abstractmethod
    def load_model(self, *args, **kwargs):
        ...

    def __call__(self, *args, **kwargs):
        image: Image.Image = self.inference_function(*args, **kwargs)
        nsfw_result = self.nsfw_classifier(image)
        print(nsfw_result, flush=True)
        if (
            nsfw_result[0]["label"] == "nsfw"
            and nsfw_result[0]["score"] > self.nsfw_threshold
        ):
            H, W = image.size
            image = Image.new("RGB", (W, H))
        return image
