import requests
import bittensor as bt
from image_generation_subnet.protocol import ImageGenerating
from typing import List


def get_prompt(self, seed: int, prompt_url: str) -> str:
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }

    data = {
        "prompt": "an image of",
        "seed": seed,
        "max_length": 42,
        "additional_params": {},
    }
    response = requests.post(prompt_url, headers=headers, json=data)
    if response.status_code != 200:
        bt.logging.error("Error getting prompt in main loop")
        raise
    prompt = response.json()["prompt"]
    return prompt


def get_reward(
    self,
    reward_url: str,
    responses: List[ImageGenerating],
    synapse: ImageGenerating,
):
    images = [response.images for response in responses]
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    data = {
        "prompt": synapse.prompt,
        "seed": synapse.seed,
        "images": images,
        "model_name": synapse.model_name,
        "additional_params": synapse.pipeline_params,
    }
    response = requests.post(reward_url, headers=headers, json=data)
    if response.status_code != 200:
        bt.logging.error("Error getting reward in main loop")
    rewards = response.json()["rewards"]
    return rewards


def get_miner_info(self, payload: dict, query_uids: List[int]):
    uid_to_axon = dict(zip(self.all_uids, self.metagraph.axons))
    query_axons = [uid_to_axon[int(uid)] for uid in query_uids]
    protocol_payload = ImageGenerating(request_dict=payload)
    bt.logging.info("Requesting miner info")
    responses = self.dendrite.query(
        query_axons,
        protocol_payload,
        deserialize=False,
        timeout=5,
    )
    bt.logging.info(f"Received {len(responses)} responses")
    responses = {
        uid: response.response_dict
        for uid, response in zip(query_uids, responses)
        if response.response_dict
    }
    bt.logging.info(f"Received {len(responses)} valid responses")

    return responses


def update_active_models(self):
    """
    1. Query model_name of available uids
    2. Update the available list
    """
    payload = {"get_miner_info": True}
    self.all_uids = [int(uid) for uid in self.metagraph.uids]
    valid_miners_info = get_miner_info(self, payload, self.all_uids)
    if not valid_miners_info:
        bt.logging.warning("No active miner available. Skipping setting weights.")
        return
    update_model_list = []
    for uid, info in valid_miners_info.items():
        if info["model_name"] in self.supporting_models:
            if uid not in self.supporting_models[info["model_name"]]["uids"]:
                self.supporting_models[info["model_name"]]["uids"].append(uid)
            update_model_list.append(info["model_name"])
