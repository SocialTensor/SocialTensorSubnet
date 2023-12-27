import requests
import time
import bittensor as bt
from image_generation_subnet.protocol import ImageGenerating
from image_generation_subnet.base.validator import BaseValidatorNeuron
import random
import torch
import os
import redis
from neurons.validator.validator_proxy import ValidatorProxy
from traceback import print_exception
from typing import List
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

REWARD_URL = os.getenv("REWARD_ENDPOINT")
PROMPT_URL = os.getenv("PROMPT_ENDPOINT")
MARKET_URL = os.getenv("MARKET_ENDPOINT")
ADMIN_GET_CONFIG_URL = os.getenv("ADMIN_GET_CONFIG_ENDPOINT")

class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()
        self.all_uids = [str(uid) for uid in self.metagraph.uids]
        self.supporting_models = {
            "RealisticVision": {
                "uids": [],
                "incentive_weight": 1.0,
                "checking_endpoint": "http://127.0.0.1:10002/verify",
            }
        }
        self.validator_proxy = ValidatorProxy(
            self.metagraph, self.dendrite, 8000, MARKET_URL
        )

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
        miners_images: List[List[str]],
        prompt: str,
        seed: int,
        model_name: str,
        additional_params: dict = {"num_inference_steps": 20},
    ):
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }
        data = {
            "prompt": prompt,
            "seed": seed,
            "images": miners_images,
            "model_name": model_name,
            "additional_params": additional_params,
        }
        response = requests.post(REWARD_URL, headers=headers, json=data)
        rewards = response.json()["rewards"]
        return rewards

    def check_miner_running_model(self, uids: List[int], model_name: str):
        """
        Generate random prompt & verify output of miner.
        """
        seed = random.randint(0, 999999)
        prompt = self.get_prompt(seed=seed)
        steps = 20
        print(prompt, uids)
        responses = self.dendrite.query(
            axons=[self.metagraph.axons[uid] for uid in uids],
            synapse=ImageGenerating(
                prompt=prompt, seed=seed, pipeline_params={"num_inference_steps": steps}
            ),
            deserialize=False,
        )
        checking_uids = []
        checking_responses = []
        for uid, response in zip(uids, responses):
            if response and response.images:
                checking_uids.append(uid)
                checking_responses.append(response)
        images = [response.images for response in checking_responses]
        rewards = self.get_reward(
            images,
            prompt,
            seed,
            model_name,
            additional_params={"num_inference_steps": steps},
        )
        valid_uids = []
        for uid, reward in zip(checking_uids, rewards):
            if reward:
                valid_uids.append(uid)
        return valid_uids

    def get_miner_info(self, payload: dict, query_uids: List[int]):
        uid_to_axon = dict(zip(self.all_uids, self.metagraph.axons))
        query_axons = [uid_to_axon[int(uid)] for uid in query_uids]
        print(query_axons)
        protocol_payload = ImageGenerating(request_dict=payload)
        print("Requesting miner info")
        responses = self.dendrite.query(
            query_axons,
            protocol_payload,
            deserialize=False,
            timeout=5,
        )
        print(f"Received {len(responses)} responses")
        responses = {
            uid: response.response_dict
            for uid, response in zip(query_uids, responses)
            if response.response_dict
        }
        print(f"Received {len(responses)} valid responses")

        return responses

    def update_active_models(self):
        """
        1. Query model_name of available uids
        2. Verify
        3. Update the available list
        """
        payload = {"get_miner_info": True}
        self.all_uids = [int(uid) for uid in self.metagraph.uids]
        print(self.all_uids)
        valid_miners_info = self.get_miner_info(payload, self.all_uids)
        if not valid_miners_info:
            bt.logging.warning("No active miner available. Skipping setting weights.")
            return
        update_model_list = []
        for uid, info in valid_miners_info.items():
            print(info)
            if (
                info["model_name"] in self.supporting_models
            ):
                if uid not in self.supporting_models[info["model_name"]]["uids"]:
                    self.supporting_models[info["model_name"]]["uids"].append(uid)
                update_model_list.append(info["model_name"])
        update_model_list = list(set(update_model_list))
        print(update_model_list)
        for model_name in update_model_list:
            self.supporting_models[model_name]["uids"] = self.check_miner_running_model(
                self.supporting_models[model_name]["uids"], model_name
            )

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        steps = 20
        seed = random.randint(0, 1000)
        prompt = self.get_prompt(seed=seed)
        model_name = random.choice(list(self.supporting_models.keys()))

        bt.logging.info(f"Received request for {model_name} model")

        bt.logging.info("Updating available models & uids")

        self.update_active_models()

        available_uids = self.supporting_models[model_name]["uids"]

        if not available_uids:
            bt.logging.warning(
                "No active miner available for specified model. Skipping setting weights."
            )
            return
        else:
            bt.logging.info(f"Available uids: {available_uids}")
        responses = self.dendrite.query(
            axons=[self.metagraph.axons[uid] for uid in available_uids],
            synapse=ImageGenerating(
                prompt=prompt, seed=seed, pipeline_params={"num_inference_steps": steps}
            ),
            deserialize=False,
        )
        valid_uids = []
        valid_responses = []
        for uid, response in zip(available_uids, responses):
            if response and response.images:
                valid_uids.append(uid)
                valid_responses.append(response)
        images = [response.images for response in valid_responses]
        rewards = self.get_reward(images, prompt, seed, model_name)
        print(rewards)
        rewards = torch.FloatTensor(rewards)
        rewards = rewards * self.supporting_models[model_name]["incentive_weight"]

        bt.logging.info(f"Scored responses: {rewards}")
        self.update_scores(rewards, valid_uids)
        self.save_state()

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that validator is registered on the network.
        self.sync()

        bt.logging.info(
            f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                bt.logging.info(f"step({self.step}) block({self.block})")

                # Run multiple forwards concurrently.
                self.loop.run_until_complete(self.concurrent_forward())
                self.validator_proxy.metagraph = self.metagraph
                self.validator_proxy.supporting_models = self.supporting_models
                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()

                self.step += 1

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error("Error during validation", str(err))
            bt.logging.debug(print_exception(type(err), err, err.__traceback__))

    def sync(self):
        """
        Wrapper for synchronizing the state of the network for the given miner or validator.
        """
        # Ensure miner or validator hotkey is still registered on the network.
        self.check_registered()

        if self.should_sync_metagraph():
            self.resync_metagraph()

        if self.should_set_weights():
            self.set_weights()
        # Always save state.
        self.save_state()


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Validator running...", time.time())
            time.sleep(5)
