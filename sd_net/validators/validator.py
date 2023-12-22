import requests
import time
import bittensor as bt
from sd_net.protocol import ImageGenerating
from template.base.validator import BaseValidatorNeuron
import random
import torch
import os
import redis
from typing import List
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_LIST = os.getenv("REDIS_LIST")
REWARD_URL = os.getenv("REWARD_ENDPOINT")
PROMPT_URL = os.getenv("PROMPT_ENDPOINT")
ADMIN_GET_CONFIG_URL = os.getenv("ADMIN_GET_CONFIG_ENDPOINT")


class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()
        self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        self.all_uids = [int(uid) for uid in self.metagraph.uids]
        self.supporting_models = {
            "sdxl-turbo-unstable-diffusers-yamermix": {
                "uids": [],
                "incentive_weight": 1.0,
                "checking_endpoint": "http://127.0.0.1:6789/verify",
                "checkpoint": "https://civitai.com/models/84040?modelVersionId=225259"
            }
        }

    def get_supporting_models(self):
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }

        response = requests.get(ADMIN_GET_CONFIG_URL, headers=headers)
        supporting_models = response.json()
        return supporting_models
    
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
        reward = response.json()["reward"]
        return reward
    
    def check_miner_running_model(self, uids: List[int]):
        '''
        Generate random prompt & verify output of miner.
        '''
        seed = random.randint(0, 999999)
        prompt = self.get_prompt(seed=seed)
        responses = self.dendrite.query(
            axons=[self.metagraph.axons[uid] for uid in uids],
            synapse=ImageGenerating(prompt=prompt, seed=seed),
            deserialize=False,
        )
        checking_uids = []
        checking_responses = []
        for uid, response in zip(uids, responses):
            if response and response.images:
                checking_uids.append(uid)
                checking_responses.append(response)

        rewards = [
            self.get_reward(response, prompt, seed) for response in checking_responses
        ]
        
        valid_uids = []
        for uid, reward in zip(checking_uids, rewards):
            if reward:
                valid_uids.append(uid)
        return valid_uids

    
    def get_miner_info(self, payload: dict, query_uids: List[int]):
        uid_to_axon = dict(zip(self.all_uids, self.metagraph.axons))
        query_axons = [uid_to_axon[int(uid)] for uid in query_uids]
        protocol_payload = ImageGenerating(info_dict=payload)
        responses = self.dendrite.query(
            query_axons,
            protocol_payload,
            deserialize = False, # All responses have the deserialize function called on them before returning. 
        )
        responses = [response['info_dict'] for response in responses]

        return responses

    def update_active_models(self):
        '''
        1. Query model_name of available uids
        2. Verify
        3. Update the available list
        '''
        payload = {'get_miner_info': True}
        self.all_uids = [int(uid) for uid in self.metagraph.uids]
        miners_info = self.get_miner_info(self.all_uids, payload)
        valid_miners_info = [(int(uid), info) for uid, info in zip(self.all_uids, miners_info) if info]
        update_model_list = []
        for uid, info in valid_miners_info:
            if info['model_name'] in self.available_models:
                self.available_models[info['model_name']]['uids'].append(uid)
                update_model_list.append(info['model_name'])
        update_model_list = list(set(update_model_list))
        for model_name in update_model_list:
            self.available_models[model_name]['uids'] = self.check_miner_running_model(self.available_models[model_name]['uids'])

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
        # enduser flow
            # item = self.redis_client.blpop(REDIS_LIST, timeout=0)
            # requested_data = eval(item[1])
            # prompt = requested_data['prompt']
            # model_name = requested_data['model_name']
        
        # testing flow
        prompt = self.get_prompt(seed=seed)
        model_name = random.choice(list(self.available_models.keys()))

        bt.logging.info(f"Received request for {model_name} model")

        bt.logging.info("Updating available models & uids")

        self.update_active_models()

        available_uids = self.available_models['model_name']['uids']

        if not available_uids:
            bt.logging.warning("No active miner available for specified model. Skipping setting weights.")
            return
        
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
        
        rewards = [
            self.get_reward(response, prompt, seed) for response in valid_responses
        ]
        rewards = torch.FloatTensor(rewards)
        rewards = rewards * self.available_models[model_name]['incentive_weight']

        bt.logging.info(f"Scored responses: {rewards}")
        self.update_scores(rewards, valid_uids)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Validator running...", time.time())
            time.sleep(5)
