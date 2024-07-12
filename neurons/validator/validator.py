import time
import os
import bittensor as bt
import random
import torch
from image_generation_subnet.base.validator import BaseValidatorNeuron
from neurons.validator.validator_proxy import ValidatorProxy
from image_generation_subnet.validator import MinerManager
import image_generation_subnet as ig_subnet
import traceback
import yaml
import threading
import math
import queue
from copy import deepcopy
from image_generation_subnet.validator.offline_challenge import (
    get_backup_image,
    get_backup_prompt,
    get_backup_llm_prompt,
)
from datetime import datetime
from neurons.core.serving_queue import QueryItem, QueryQueue


class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        bt.logging.info("load_state()")
        self.miner_manager = MinerManager(self)
        self.load_state()
        self.update_scores_on_chain()
        self.sync()
        self.miner_manager.update_miners_identity()
        self.query_queue = QueryQueue(
            list(self.nicheimage_catalogue.keys()),
            time_per_loop=self.config.loop_base_time,
        )
        if self.config.proxy.port:
            try:
                self.validator_proxy = ValidatorProxy(self)
                bt.logging.info("Validator proxy started succesfully")
            except Exception:
                bt.logging.warning(
                    "Warning, proxy did not start correctly, so no one can query through your validator. Error message: "
                    + traceback.format_exc()
                )

    def forward(self):
        """
        Validator synthetic forward pass. Consists of:
        - Querying all miners to get their model_name and total_volume
        - Create serving queue, here is pseudo code:
            ```
                synthentic_queue = Queue()
                for uid, total_volume_this_validator in all_uids_info:
                    for _ in range(total_volume_this_validator*0.8):
                        synthentic_queue.put(uid)
                shuffle(synthentic_queue)

                organic_queue = Queue()
                for uid, total_volume_this_validator in all_uids_info:
                    for _ in range(total_volume_this_validator*0.2):
                        organic_queue.put(uid)
                shuffle(organic_queue)
            ```
        - Forwarding requests to miners in multiple thread to ensure total time is around 600 seconds. In each thread, we do:
            - Calculating rewards if needed
            - Updating scores based on rewards
            - Saving the state
        - Normalize weights based on incentive_distribution
        - SET WEIGHTS!
        - Sleep for 600 seconds if needed
        """

        bt.logging.info("Updating available models & uids")
        async_batch_size = self.config.async_batch_size
        loop_base_time = self.config.loop_base_time  # default is 600 seconds
        threads = []
        loop_start = time.time()
        self.miner_manager.update_miners_identity()
        self.query_queue.update_queue(self.miner_manager.all_uids_info)
        for (
            model_name,
            uids,
            should_rewards,
            sleep_per_batch,
        ) in self.query_queue.get_batch_query(async_batch_size):
            bt.logging.info(
                f"Querying {len(uids)} uids for model {model_name}, sleep_per_batch: {sleep_per_batch}"
            )

            thread = threading.Thread(
                target=self.async_query_and_reward,
                args=(model_name, uids, should_rewards),
            )
            threads.append(thread)
            thread.start()

            bt.logging.info(f"Sleeping for {sleep_per_batch} seconds between batches")
            time.sleep(sleep_per_batch)

        for thread in threads:
            thread.join()
        self.update_scores_on_chain()
        self.save_state()
        bt.logging.info(
            "Loop completed, uids info:\n",
            str(self.miner_manager.all_uids_info).replace("},", "},\n"),
        )

        actual_time_taken = time.time() - loop_start

        if actual_time_taken < loop_base_time:
            bt.logging.info(
                f"Sleeping for {loop_base_time - actual_time_taken} seconds"
            )
            time.sleep(loop_base_time - actual_time_taken)

    def async_query_and_reward(
        self,
        model_name: str,
        uids: list[int],
        should_rewards: list[int],
    ):
        dendrite = bt.dendrite(self.wallet)
        if model_name == "RealitiesEdgeXL" and datetime.utcnow() < datetime(2024, 6, 12, 0, 0, 0):
            pipeline_type = "txt2img"
        else:
            pipeline_type = random.choice(
                self.nicheimage_catalogue[model_name]["supporting_pipelines"]
            )
        reward_url = self.nicheimage_catalogue[model_name]["reward_url"]
        uids_should_rewards = list(zip(uids, should_rewards))
        synapses, batched_uids_should_rewards = self.prepare_challenge(
            uids_should_rewards, model_name, pipeline_type
        )
        for synapse, uids_should_rewards in zip(synapses, batched_uids_should_rewards):
            uids, should_rewards = zip(*uids_should_rewards)
            bt.logging.info(f"Quering {uids}, Should reward: {should_rewards}")
            if not synapse:
                continue
            base_synapse = synapse.copy()
            axons = [self.metagraph.axons[int(uid)] for uid in uids]
            responses = dendrite.query(
                axons=axons,
                synapse=synapse,
                deserialize=False,
                timeout=self.nicheimage_catalogue[model_name]["timeout"],
            )
            reward_responses = [
                response
                for response, should_reward in zip(responses, should_rewards)
                if should_reward
            ]
            reward_uids = [
                uid for uid, should_reward in zip(uids, should_rewards) if should_reward
            ]

            bt.logging.info(
                f"Received {len(responses)} responses, {len(reward_responses)} to be rewarded"
            )
            store_thread = threading.Thread(
                target=self.store_miner_output,
                args=(self.config.storage_url, responses, uids, self.uid),
                daemon=True,
            )
            store_thread.start()

            if reward_uids:
                if callable(reward_url):
                    reward_uids, rewards = reward_url(
                        base_synapse, reward_responses, reward_uids
                    )
                else:
                    reward_uids, rewards = ig_subnet.validator.get_reward(
                        reward_url,
                        base_synapse,
                        reward_responses,
                        reward_uids,
                        self.nicheimage_catalogue[model_name].get("timeout", 12),
                        self.miner_manager,
                    )

                    # Scale Reward based on Miner Volume
                for i, uid in enumerate(reward_uids):
                    if rewards[i] > 0:
                        rewards[i] = rewards[i] * (
                            0.6 + 0.4 * self.miner_manager.all_uids_info[uid]["reward_scale"]
                        )

                bt.logging.info(f"Scored responses: {rewards}")

                self.miner_manager.update_scores(reward_uids, rewards)
            store_thread.join()

    def prepare_challenge(self, uids_should_rewards, model_name, pipeline_type):
        synapse_type = self.nicheimage_catalogue[model_name]["synapse_type"]
        model_miner_count = len(
            [
                uid
                for uid, info in self.miner_manager.all_uids_info.items()
                if info["model_name"] == model_name
            ]
        )
        batch_size = min(4, 1 + model_miner_count // 4)
        random.shuffle(uids_should_rewards)
        batched_uids_should_rewards = [
            uids_should_rewards[i * batch_size : (i + 1) * batch_size]
            for i in range((len(uids_should_rewards) + batch_size - 1) // batch_size)
        ]
        num_batch = len(batched_uids_should_rewards)
        synapses = [
            synapse_type(pipeline_type=pipeline_type, model_name=model_name)
            for _ in range(num_batch)
        ]
        for synapse in synapses:
            synapse.pipeline_params.update(
                self.nicheimage_catalogue[model_name]["inference_params"]
            )
            synapse.seed = random.randint(0, 1e9)
        for challenge_url, backup_func in zip(
            self.challenge_urls[pipeline_type]["main"],
            self.challenge_urls[pipeline_type]["backup"],
        ):
            if callable(challenge_url):
                synapses = challenge_url(synapses)
            else:
                assert isinstance(challenge_url, str)
                synapses = ig_subnet.validator.get_challenge(
                    challenge_url, synapses, backup_func
                )
        return synapses, batched_uids_should_rewards

    def store_miner_output(
        self, storage_url, responses: list[bt.Synapse], uids, validator_uid
    ):
        if not self.config.share_response:
            return
        
        for uid, response in enumerate(responses):
            if not response.is_success:
                continue
            try:
                response.store_response(storage_url, uid, validator_uid)
                break
            except Exception as e:
                bt.logging.error(f"Error in storing response: {e}")

    def update_scores_on_chain(self):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        weights = torch.zeros(len(self.miner_manager.all_uids))
        for model_name in self.nicheimage_catalogue.keys():
            model_specific_weights = self.miner_manager.get_model_specific_weights(
                model_name
            )
            
            # Smoothing update incentive
            temp_incentive_weight = {}
            if datetime.utcnow() < datetime(2024, 6, 6, 0, 0, 0):
                temp_incentive_weight = {
                    "DallE": 0.01,
                    "AnimeV3": 0.30,
                }

            if model_name in temp_incentive_weight:
                bt.logging.info(f"Using temp_incentive_weight: {temp_incentive_weight} for {model_name}")
                model_specific_weights = (
                    model_specific_weights * temp_incentive_weight[model_name]
                )
            else:
                model_specific_weights = (
                    model_specific_weights
                    * self.nicheimage_catalogue[model_name]["model_incentive_weight"]
                )
            bt.logging.info(
                f"model_specific_weights for {model_name}\n{model_specific_weights}"
            )
            weights = weights + model_specific_weights

        # Check if rewards contains NaN values.
        if torch.isnan(weights).any():
            bt.logging.warning(f"NaN values detected in weights: {weights}")
            # Replace any NaN values in rewards with 0.
            weights = torch.nan_to_num(weights, 0)
        self.scores: torch.FloatTensor = weights
        bt.logging.success(f"Updated scores: {self.scores}")

    def save_state(self):
        """Saves the state of the validator to a file."""

        torch.save(
            {
                "step": self.step,
                "all_uids_info": self.miner_manager.all_uids_info,
            },
            self.config.neuron.full_path + "/state.pt",
        )

    def load_state(self):
        """Loads the state of the validator from a file."""

        # Load the state of the validator from file.
        try:
            path = self.config.neuron.full_path + "/state.pt"
            bt.logging.info("Loading validator state from: " + path)
            state = torch.load(path)
            self.step = state["step"]
            self.miner_manager.all_uids_info = state["all_uids_info"]
            bt.logging.info("Succesfully loaded state")
        except Exception as e:
            self.step = 0
            bt.logging.info("Could not find previously saved state.", e)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Validator running...", time.time())
            time.sleep(360)
