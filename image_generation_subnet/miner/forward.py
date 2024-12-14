from typing import Union
import requests
import httpx
import bittensor as bt
import torch

from image_generation_subnet.protocol import ImageGenerating, TextGenerating, MultiModalGenerating

try:
    GPU_DEVICE_NAME = torch.cuda.get_device_name()
    GPU_DEVICE_COUNT = torch.cuda.device_count()
except Exception:
    GPU_DEVICE_NAME = "cpu"
    GPU_DEVICE_COUNT = 0


def set_info(self) -> dict:
    """
    Returns miner information, which can vary depending on the layer type (layer zero or layer one).

    The returned dictionary may have different structures based on the context. Examples:

    #### For Layer One:
    {
        "layer_one": {
            "ip": "127.0.0.1",
            "port": 8080
        },
        "is_layer_zero": True
    }

    #### For Layer Zero (or other configurations):
    {
        "model_name": "Llama3_70b",
        "total_volume": 1000,
        "size_preference_factor": 1.0,
        "min_stake": 100,
        "volume_per_validator": {
            0: 100,
            1: 200,
            2: 300
        },
        "device_info": {
            "gpu_device_name": "A100",
            "gpu_device_count": 8
        },
        "is_layer_zero": False,
        "is_layer_one": True
    }

    Returns:
        dict: A dictionary containing miner configuration data, which may include information about the miner's layer, 
              model, volume per validator, device information, and other configuration details.
    """

    if self.config.miner.is_layer_zero:
        miner_info = {
            "layer_one": {
                "ip": self.config.miner.layer_one_ip,
                "port": self.config.miner.layer_one_port,
            },
            "is_layer_zero": True,
        }
    else:
        is_layer_one = True if self.config.miner.is_layer_one else False
        response = get_model_name(self)
        miner_info = {
            "model_name": response["model_name"],
            "total_volume": self.config.miner.total_volume,
            "size_preference_factor": self.config.miner.size_preference_factor,
            "min_stake": self.config.miner.min_stake,
            "volume_per_validator": self.volume_per_validator,
            "device_info": {
                "gpu_device_name": GPU_DEVICE_NAME,
                "gpu_device_count": GPU_DEVICE_COUNT,
            },
            "is_layer_zero": False,
            "is_layer_one": is_layer_one
        }
    return miner_info


async def generate(self, synapse: Union[ImageGenerating, TextGenerating, MultiModalGenerating]) -> ImageGenerating | TextGenerating | MultiModalGenerating:
    """
    Deserialize the synapse and send request to the generate endpoint.
    """
    data = synapse.deserialize_input()
    data["timeout"] = synapse.timeout
    async with httpx.AsyncClient() as client:
        response = await client.post(
            self.config.generate_endpoint, json=data, timeout=synapse.timeout
        )
    if response.status_code != 200:
        raise Exception(f"Error in generate: {response.json()}")
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
