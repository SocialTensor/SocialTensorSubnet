import requests
import httpx
import bittensor as bt
import torch
from image_generation_subnet.utils.commit_on_chain import compress_dict, decompress_dict

try:
    GPU_DEVICE_NAME = torch.cuda.get_device_name()
    GPU_DEVICE_COUNT = torch.cuda.device_count()
except Exception:
    GPU_DEVICE_NAME = "cpu"
    GPU_DEVICE_COUNT = 0

def set_info(self):
    # Set information of miner
    # Currently only model name is set
    response = get_model_name(self)
    miner_info = {
        "model_name": response["model_name"],
        "total_volume": self.config.miner.total_volume,
        "device_info": {
            "name": GPU_DEVICE_NAME,
            "count": GPU_DEVICE_COUNT,
        }
    }
    return miner_info


async def generate(self, synapse: bt.Synapse) -> bt.Synapse:
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

