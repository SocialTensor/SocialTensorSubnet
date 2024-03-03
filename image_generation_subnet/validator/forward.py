import requests
import bittensor as bt
from image_generation_subnet.protocol import ImageGenerating
from typing import List
from math import pow
from functools import wraps
from tqdm import tqdm

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
    url: str, synapses: List[ImageGenerating]
) -> List[ImageGenerating]:
    for i, synapse in tqdm(enumerate(synapses), total=len(synapses)):
        if not synapse:
            continue
        try:
            data = synapse.deserialize()
            response = requests.post(url, json=data)
            if response.status_code != 200:
                raise
            challenge = response.json()
        except Exception:
            challenge = None
        if challenge:
            synapses[i] = synapse.copy(update=challenge)
        else:
            synapses[i] = None
    return synapses


def get_reward(
    url: str,
    base_synapse: ImageGenerating,
    synapses: List[ImageGenerating],
    uids: List[int],
) -> List[float]:
    valid_uids = [uid for uid, response in zip(uids, synapses) if response.is_success]
    invalid_uids = [
        uid for uid, synapse in zip(uids, synapses) if not synapse.is_success
    ]
    total_uids = valid_uids + invalid_uids
    valid_synapses = [synapse for synapse in synapses if synapse.is_success]
    if valid_uids:
        data = {
            "miner_data": [synapse.deserialize() for synapse in valid_synapses],
            "base_data": base_synapse.deserialize(),
        }
        response = requests.post(url, json=data)
        if response.status_code != 200:
            raise Exception(f"Error in get_reward: {response.json()}")
        valid_rewards = response.json()["rewards"]
        valid_rewards = [float(reward) for reward in valid_rewards]
        process_times = [synapse.dendrite.process_time for synapse in valid_synapses]

        valid_rewards = add_time_penalty(valid_rewards, process_times)
        valid_rewards = [round(num, 3) for num in valid_rewards]
    else:
        bt.logging.info("0 valid responses in a batch")
        valid_rewards = []

    total_rewards = valid_rewards + [0] * len(invalid_uids)

    return total_uids, total_rewards

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
