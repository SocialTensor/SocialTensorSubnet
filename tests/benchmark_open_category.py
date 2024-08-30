from neurons.validator.validator import initialize_nicheimage_catalogue, Validator
from image_generation_subnet.protocol import ImageGenerating
import argparse
import yaml
import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor
import random
from image_generation_subnet.utils.config import config, add_args
import bittensor as bt

def get_catalogue():
    parser = argparse.ArgumentParser()
    add_args(Validator, parser)

    CONFIG = bt.config(parser)
    return initialize_nicheimage_catalogue(CONFIG)

CONFIG = yaml.load(open("generation_models/configs/model_config.yaml"), yaml.FullLoader)
MODEL_NAME = "OpenCategory"

parser = argparse.ArgumentParser(description='Benchmark OpenCategory')
parser.add_argument('--n_times', type=int, default=5, help='Number of times to benchmark')
parser.add_argument("--max_concurrent_requests", type=int, default=1, help="Number of concurrent requests")
parser.add_argument(
    "--generate_endpoint",
    type=str,
    help="The endpoint to send generate requests to.",
    default="http://127.0.0.1:10006/generate",
)
parser.add_argument(
    "--challenge_prompt",
    type=str,
    help="The endpoint to create prompt challenge",
    default="http://nicheimage.nichetensor.com/challenge/prompt",
)

args = parser.parse_args()
model_catalogue = get_catalogue()
MODEL_CONFIG = CONFIG[MODEL_NAME]

print(f"Model catalogue: {model_catalogue[MODEL_NAME]}")

synapse = ImageGenerating(model_name=MODEL_NAME, timeout=model_catalogue[MODEL_NAME]["timeout"])
synapse.prompt = ""
synapse.seed = random.randint(0, 1000000)
synapse.pipeline_params = model_catalogue[MODEL_NAME]["inference_params"]

def benchmark_open_category(n_times, model_name, n_concurrent_requests, generate_endpoint):
    # report times, status codes and rewards
    import time
    def _post(synapse: ImageGenerating):
        result = None
        timeout = synapse.timeout
        synapse.pipeline_type = "txt2img"
        print(synapse.pipeline_type)
        start = time.time()
        try:
            response = requests.post(generate_endpoint, json=synapse.deserialize_input(), timeout=timeout)
        except requests.exceptions.ReadTimeout:
            return result, 408, timeout
        end = time.time()

        if response.status_code == 200:
            result = response.json()
        return result, response.status_code, end - start
    
    def _get_prompt_challenge(url, sleep_time = 60):
        prompt = None
        while not prompt:
            response = requests.post(url, json={})
            if response.status_code == 200:
                prompt = response.json()["prompt"]
            else:
                
                print(f"Rate Limit Exceeded! Sleeping for {sleep_time} sec due to rate limit on challenge prompt server !")
                time.sleep(sleep_time)
        # time.sleep(sleep_time)
        return prompt

    def _get_reward(base_synapse, synapses, url, sleep_time = 60):
        rewards= None
        while not rewards:
            data = {
                "miner_data": [synapse.deserialize() for synapse in synapses],
                "base_data": base_synapse.deserialize_input(),
            }
            response = requests.post(url, json=data)
            if response.status_code != 200:
                print(f"Error in get_reward: {response.text}")
                print(f"Rate Limit Exceeded! Sleeping for {sleep_time} sec until query reward server again!")
                time.sleep(sleep_time)
            else:
                rewards = response.json()
        return rewards

    reports = {}
    rewards = {}
    print(f"Benchmarking {model_name} with {n_times} times and {n_concurrent_requests} concurrent requests")
    print("Starting benchmark...")
    for _ in tqdm.tqdm(range(n_times)):
        prompt = _get_prompt_challenge(args.challenge_prompt)
        if prompt:
            synapse.prompt = prompt
            base_synapses = synapse.copy()
            if prompt not in rewards:
                rewards[prompt] = {
                    "base_synapse": base_synapses,
                    "synapses": [],
                    "score": []
                }
            with ThreadPoolExecutor(max_workers=n_concurrent_requests) as executor:
                futures = []
                for _ in range(n_concurrent_requests):
                    futures.append(executor.submit(_post, synapse))
                for future in futures:
                    result, status_code, time_taken = future.result()
                    if status_code not in reports:
                        reports[status_code] = []
                    reports[status_code].append(time_taken)
                    
                    if result:
                        synapse.image = result["image"]
                        rewards[prompt]["synapses"].append(synapse)
        

    for prompt, data in rewards.items():
        reward = _get_reward(data["base_synapse"], data["synapses"], model_catalogue[MODEL_NAME]["reward_url"])
        data["score"].append(reward)
        data.pop("base_synapse", None)
        data.pop("synapses", None)
    
    return reports, rewards

if __name__ == "__main__":
    reports, rewards = benchmark_open_category(args.n_times, MODEL_NAME, args.max_concurrent_requests, args.generate_endpoint)
    print("[Status code and process time] ", reports)
    print("[Reward] ", rewards)
