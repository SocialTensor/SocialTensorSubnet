from image_generation_subnet.protocol import ImageGenerating
import random


def get_promptGoJouney(synapses: list[ImageGenerating]) -> list[ImageGenerating]:
    ars = [
        "16:9",
        "1:1",
        "9:16",
        "4:5",
        "5:4",
        "3:4",
        "4:3",
        "2:3",
        "3:2",
    ]
    for synapse in synapses:
        synapse.prompt = f"{synapse.prompt} --ar {random.choice(ars)} --v 6"
    return synapses
