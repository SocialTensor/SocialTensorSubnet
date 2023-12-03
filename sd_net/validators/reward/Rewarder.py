from typing import List
from PIL import Image
import torch
import torch.nn as nn


class Rewarder(nn.Module):
    def __init__(self):
        self.model = None

    def forward(self, prompt: str, images: List[List[Image.Image]]):
        # TODO(Develop): write algorithm for verify "Is images generated from Model(model_name)?"
        return torch.FloatTensor([1 for _ in range(len(images))])
