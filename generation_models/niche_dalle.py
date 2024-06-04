from .base_model import BaseModel
import openai
import os
from diffusers.utils import load_image
import random
from openai import OpenAI

URL_REGEX = (
    r'https://(?:oaidalleapiprodscus|dalleprodsec)\.blob\.core\.windows\.net/private/org-[\w-]+/'
    r'user-[\w-]+/img-[\w-]+\.(?:png|jpg)\?'
    r'st=\d{4}-\d{2}-\d{2}T\d{2}%3A\d{2}%3A\d{2}Z&'
    r'se=\d{4}-\d{2}-\d{2}T\d{2}%3A\d{2}%3A\d{2}Z&'
    r'(?:sp=\w+&)?'
    r'sv=\d{4}-\d{2}-\d{2}&'
    r'sr=\w+&'
    r'rscd=\w+&'
    r'rsct=\w+/[\w-]+&'
    r'skoid=[\w-]+&'
    r'sktid=[\w-]+&'
    r'skt=\d{4}-\d{2}-\d{2}T\d{2}%3A\d{2}%3A\d{2}Z&'
    r'ske=\d{4}-\d{2}-\d{2}T\d{2}%3A\d{2}%3A\d{2}Z&'
    r'sks=\w+&'
    r'skv=\d{4}-\d{2}-\d{2}&'
    r'sig=[\w/%+=]+'
)




class NicheDallE(BaseModel):
    def __init__(self, *args, **kwargs):
        self.inference_function = self.load_model(*args, **kwargs)
        self.client = OpenAI()

    def load_model(self, *args, **kwargs):
        imagine_inference_function = self.load_imagine(*args, **kwargs)
        return imagine_inference_function

    def __call__(self, *args, **kwargs):
        return self.inference_function(*args, **kwargs)

    def load_imagine(self, *args, **kwargs):
        supporting_sizes = ["1792x1024", "1024x1792"]
        def inference_function(*args, **kwargs):
            prompt = kwargs.get("prompt", "a cute cat")
            style = kwargs.get("style", "natural")
            if style not in ["vivid", "natural"]:
                style = "natural"
            size = kwargs.get("size", random.choice(supporting_sizes))
            if size not in supporting_sizes:
                size = random.choice(supporting_sizes)
            response_obj = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size=size,
                response_format="url",
                style=style
            )
            data = {
                "url": response_obj.data[0].url,
                "revised_prompt": response_obj.data[0].revised_prompt
            }
            print(data, flush=True)
            return data

        return inference_function
