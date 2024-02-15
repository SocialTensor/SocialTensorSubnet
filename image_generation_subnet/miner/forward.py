import requests
from typing import List
import bittensor as bt
from image_generation_subnet.protocol import NicheImageProtocol


def set_info(self):
    # Set information of miner
    # Currently only model name is set
    miner_info = {}
    response = get_model_name(self)
    miner_info["model_name"] = response["model_name"]
    miner_info["category"] = response["category"]
    return miner_info


def generate(self, synapse: NicheImageProtocol) -> NicheImageProtocol:
    data = synapse.deserialize()

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }

    response = requests.post(self.config.generate_endpoint, headers=headers, json=data)
    synapse = synapse.copy(update=response.json())
    return synapse


def get_model_name(self):
    # Get running model's name
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    response = requests.get(self.config.info_endpoint, headers=headers)
    response = response.json()
    return response
