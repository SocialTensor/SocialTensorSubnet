from image_generation_subnet.protocol import ImageGenerating
import requests


def fetch_GoJourney(task_id):
    endpoint = "https://api.midjourneyapi.xyz/mj/v2/fetch"
    data = {"task_id": task_id}
    response = requests.post(endpoint, json=data)
    return response.json()


def get_reward_GoJourney(
    base_synapse: ImageGenerating,
    synapses: list[ImageGenerating],
    uids: list,
    *args,
    **kwargs,
) -> list[float]:
    reward_distribution = {
        "turbo": 0.14,
        "fast": 0.07,
        "relax": 0.03,
    }
    rewards = []
    prompt = base_synapse.prompt
    for synapse in synapses:
        try:
            synapse_response = synapse.response_dict
            task_id = synapse_response["task_id"]
            task_request = synapse_response["task_request"]
            task_response = fetch_GoJourney(task_id)
            task_status = task_response["status"]
            if task_status == "failed" or task_request["prompt"] != prompt:
                reward = 0
            else:
                process_mode = task_request["process_mode"]
                reward = reward_distribution[process_mode]
            rewards.append(reward)
        except Exception:
            rewards.append(0)
    return uids, rewards