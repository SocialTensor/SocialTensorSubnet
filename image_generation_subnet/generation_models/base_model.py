from abc import ABC, abstractmethod
from transformers import pipeline
from PIL import Image


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
        nsfw_probability = nsfw_result[1]["score"]
        if nsfw_probability > self.nsfw_threshold:
            H, W = image.size
            image = Image.new("RGB", (W, H))
        return image
