import requests
import bittensor as bt
from image_generation_subnet.protocol import ImageGenerating
from typing import List
from functools import wraps


def retry(**kwargs):
    module = kwargs.get("module", "unknown")

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # retry forever
            while True:
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    bt.logging.error(f"Error in {module}: {e}, retrying...")

        return wrapper

    return decorator


def skip(**kwargs):
    module = kwargs.get("module", "unknown")

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # return None if error
            try:
                return f(*args, **kwargs)
            except Exception as e:
                bt.logging.error(f"Error in {module}: {e}, skipping...")
                return None

        return wrapper

    return decorator


@skip(module="prompting")
def get_prompt(seed: int, prompt_url: str) -> str:
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
    prompt = response.json()["prompt"]
    return prompt


@skip(module="rewarding")
def get_reward(
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
    model_uids_dict = {}
    for uid, info in valid_miners_info.items():
        if info["model_name"] in self.supporting_models:
            if info["model_name"] not in model_uids_dict:
                model_uids_dict[info["model_name"]] = []
            model_uids_dict[info["model_name"]].append(uid)
    for model_name, uids in model_uids_dict.items():
        self.supporting_models[model_name]["uids"] = uids
