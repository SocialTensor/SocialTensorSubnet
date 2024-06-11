import torch
from transformers import pipeline


class Moderation:
    def __init__(self):
        self.moderation_model = pipeline(
            "text-classification",
            model="AdamCodd/distilroberta-nsfw-prompt-stable-diffusion",
        )

    @torch.inference_mode()
    def __call__(self, prompt, threshold=0.5):
        result = self.moderation_model(prompt)
        if result[0]["label"] == "NSFW" and result[0]["score"] > threshold:
            return True, (prompt, result)
        return False, (prompt, result)
