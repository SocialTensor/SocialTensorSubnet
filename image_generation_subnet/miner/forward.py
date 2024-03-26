import requests
import httpx
import bittensor as bt

def set_info(self):
    # Set information of miner
    # Currently only model name is set
    response = get_model_name(self)
    miner_info = {
        "model_name": response["model_name"],
        "total_volume": self.config.miner.total_volume,
        "size_preference_factor": self.config.miner.size_preference_factor,
        "min_stake": self.config.miner.min_stake,
        "volume_per_validator": self.volume_per_validator,
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

