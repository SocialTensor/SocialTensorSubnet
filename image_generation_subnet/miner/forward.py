import requests
import httpx
import bittensor as bt
import torch

try:
    GPU_DEVICE_NAME = torch.cuda.get_device_name()
    GPU_DEVICE_COUNT = torch.cuda.device_count()
except Exception:
    GPU_DEVICE_NAME = "cpu"
    GPU_DEVICE_COUNT = 0


def set_info(self):
    if self.config.miner.is_layer_zero:
        miner_info = {
            "layer_one": {
                "ip": self.config.miner.layer_one_ip,
                "port": self.config.miner.layer_one_port,
            }
        }
    else:
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
