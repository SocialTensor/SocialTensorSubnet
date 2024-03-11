from image_generation_subnet.protocol import ImageGenerating
import requests
import bittensor as bt


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
        "turbo": 1.0,
        "fast": 0.5,
        "relax": 0.1,
    }
    rewards = []
    prompt = base_synapse.prompt
    for synapse in synapses:
        try:
            synapse_response: dict = synapse.response_dict
            bt.logging.info(synapse_response)
            task_id = synapse_response["task_id"]
            task_response = fetch_GoJourney(task_id)
            task_request = task_response["meta"]["task_request"]
            task_status = task_response["status"]
            bt.logging.info(f"Synapse base: {base_synapse}")
            bt.logging.info(f"Task status: {task_status}")
            bt.logging.info(f"Task request: {task_request}")
            bt.logging.info(f"Task response: {task_response}")
            if task_status == "failed":
                bt.logging.info("Task failed")
                reward = 0
            elif task_request["prompt"].split("--")[0] != prompt.split("--")[0]:
                bt.logging.info(
                    f"Prompt mismatch: {task_request['prompt']} != {prompt}"
                )
                reward = 0
            else:
                process_mode = task_response["meta"]["task_request"]["process_mode"]
                reward = reward_distribution[process_mode]
                bt.logging.info(f"Process_mode: {process_mode}")
            rewards.append(reward)
        except Exception as e:
            bt.logging.warning(f"Error in get_reward_GoJourney: {e}")
            rewards.append(0)
    return uids, rewards
