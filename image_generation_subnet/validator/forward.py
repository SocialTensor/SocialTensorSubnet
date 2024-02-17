import requests
import bittensor as bt
from image_generation_subnet.protocol import NicheImageProtocol
from typing import List
from math import pow
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


def get_challenge(
    url: str, synapses: List[NicheImageProtocol]
) -> List[NicheImageProtocol]:
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    datas = [synapse.deserialize() for synapse in synapses]
    challenges = []
    for data in datas:
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code != 200:
                raise Exception(f"Error in get_challenge: {response.json()}")
            challenge = response.json()
        except Exception as e:
            bt.logging.error(f"Error in get_challenge: {e}")
            challenge = None
        challenges.append(challenge)
    synapses = [
        synapse.copy(update=challenge)
        for synapse, challenge in zip(synapses, challenges)
    ]
    return synapses


def get_reward(
    url: str, base_synapse: NicheImageProtocol, synapes: List[NicheImageProtocol]
) -> List[float]:
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    if not synapes:
        return []
    data = {
        "miner_data": [synapse.deserialize() for synapse in synapes],
        "base_data": base_synapse.deserialize(),
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"Error in get_reward: {response.json()}")
    rewards = response.json()["rewards"]
    rewards = [float(reward) for reward in rewards]

    return rewards

def get_miner_info(validator, query_uids: List[int]):
    uid_to_axon = dict(zip(validator.all_uids, validator.metagraph.axons))
    query_axons = [uid_to_axon[int(uid)] for uid in query_uids]
    synapse = NicheImageProtocol()
    synapse.request_dict = {"get_miner_info": True}
    bt.logging.info("Requesting miner info")
    responses = validator.dendrite.query(
        query_axons,
        synapse,
        deserialize=True,
        timeout=10,
    )
    responses = {
        uid: response
        for uid, response in zip(query_uids, responses)
        if response and "model_name" in response and "category" in response
    }
    return responses


def update_active_models(validator):
    """
    1. Query model_name of available uids
    2. Update the available list
    """
    miner_distribution = {}
    validator.all_uids = [int(uid) for uid in validator.metagraph.uids]
    valid_miners_info = get_miner_info(validator, validator.all_uids)
    if not valid_miners_info:
        bt.logging.warning("No active miner available. Skipping setting weights.")
    for uid, info in valid_miners_info.items():
        miner_state = validator.all_uids_info.setdefault(
            uid, {"scores": [], "model_name": "", "category": ""}
        )
        model_name = info["model_name"]
        category = info["category"]
        miner_distribution.setdefault(f"{category}-{model_name}", 0)
        miner_distribution[f"{category}-{model_name}"] += 1
        if (
            miner_state["model_name"] == model_name
            and miner_state["category"] == category
        ):
            continue
        miner_state["model_name"] = model_name
        miner_state["category"] = category
        miner_state["scores"] = []
    bt.logging.success(f"Updated miner distribution: {miner_distribution}")


def add_time_penalty(rewards, process_times, max_penalty=0.4):
    """
    Add time penalty to rewards, based on process time
    """
    penalties = [
        max_penalty * pow(process_time, 3) / pow(12, 3)
        for process_time in process_times
    ]
    penalties = [min(penalty, max_penalty) for penalty in penalties]
    for i in range(len(rewards)):
        if rewards[i] > 0:
            rewards[i] = rewards[i] - penalties[i]
    return rewards
