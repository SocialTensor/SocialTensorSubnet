import requests
import time
import bittensor as bt
from sd_net.validators.utils.uids import get_random_uids
from sd_net.protocol import ImageGenerating, pil_image_to_base64
from template.base.validator import BaseValidatorNeuron
import random
import torch
import os
import redis
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_LIST = os.getenv("REDIS_LIST")
REWARD_URL = os.getenv("REWARD_ENDPOINT")
PROMPT_URL = os.getenv("PROMPT_ENDPOINT")


class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()
        # TODO(developer): Anything specific to your use case you can do here
        self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

    def get_prompt(self, seed: int) -> str:
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }

        data = {
            "prompt": "an image of",
            "seed": seed,
            "max_length": 77,
            "additional_params": {},
        }

        response = requests.post(PROMPT_URL, headers=headers, json=data)
        prompt = response.json()["prompt"]
        return prompt

    def get_reward(
        self,
        miner_response: ImageGenerating,
        prompt: str,
        seed: int,
        additional_params: dict = {},
    ):
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }
        miner_images = miner_response.images
        data = {
            "prompt": prompt,
            "seed": seed,
            "images": miner_images,
            "additional_params": additional_params,
        }
        response = requests.post(REWARD_URL, headers=headers, json=data)
        print(response)
        reward = response.json()["reward"]
        return reward

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        seed = random.randint(0, 1000)
        item = self.redis_client.blpop(REDIS_LIST, timeout=0)
        requested_data = eval(item[1])
        prompt = requested_data['prompt']
        print(prompt)
        available_uids = get_random_uids(self, k=self.config.neuron.sample_size)
        print(f"UIDS: {available_uids}")
        responses = self.dendrite.query(
            axons=[self.metagraph.axons[uid] for uid in available_uids],
            synapse=ImageGenerating(prompt=prompt, seed=seed),
            deserialize=False,
        )
        valid_uids = []
        valid_responses = []
        for uid, response in zip(available_uids, responses):
            if response and response.images:
                valid_uids.append(uid)
                valid_responses.append(response)

        # bt.logging.info(f"Received responses: {valid_responses}")

        rewards = [
            self.get_reward(response, prompt, seed) for response in valid_responses
        ]
        rewards = torch.FloatTensor(rewards)
        bt.logging.info(f"Scored responses: {rewards}")
        self.update_scores(rewards, valid_uids)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Validator running...", time.time())
            time.sleep(5)
