from image_generation_subnet.protocol import ImageGenerating
import random
import requests
import bittensor as bt


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
    synapses = check_batch_prompt(synapses)
    for synapse in synapses:
        synapse.prompt = f"{synapse.prompt} --ar {random.choice(ars)} --v 6"
    return synapses


def check_a_prompt(prompt: str) -> str:
    endpoint = "https://api.midjourneyapi.xyz/mj/v2/validation"
    data = {"prompt": prompt}
    response = requests.post(endpoint, json=data, timeout=10)
    response = response.json()
    return response["ErrorMessage"]


def get_offline_prompt():
    landscapes = ["landscape", "mountain", "forest", "beach", "desert", "city"]
    animals = ["dog", "cat", "bird", "fish", "horse", "rabbit"]
    actions = ["running", "jumping", "flying", "swimming", "sitting", "standing"]
    return f"{random.choice(landscapes)} with {random.choice(animals)} {random.choice(actions)}, {random.randint(1, 10000)}"


def check_batch_prompt(synapses: list[ImageGenerating]) -> list[ImageGenerating]:
    for synapse in synapses:
        if not synapse:
            continue
        if check_a_prompt(synapse.prompt):
            synapse.prompt = get_offline_prompt()
            bt.logging.warning(
                f"Prompt {synapse.prompt} is not valid, use offline prompt: {synapse.prompt}"
            )
    return synapses
