import requests
from typing import List
import bittensor as bt


def set_info(self):
    # Set information of miner
    # Currently only model name is set
    miner_info = {}
    model_name = get_model_name(self)
    miner_info["model_name"] = model_name
    return miner_info


def generate(self, prompt: str, seed: int, additional_params: dict) -> List[str]:
    data = {"prompt": prompt, "seed": seed, "additional_params": additional_params}

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }

    response = requests.post(self.config.generate_endpoint, headers=headers, json=data)
    image = response.json()["image"]
    return image


def get_model_name(self):
    # Get running model's name
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    response = requests.get(self.config.info_endpoint, headers=headers)
    model_name = response.json()["model_name"]
    return model_name
