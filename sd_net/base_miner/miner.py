import time
import typing
import bittensor as bt
from template.base.miner import BaseMinerNeuron
import sd_net
import torch
from sd_net.protocol import pil_image_to_base64
from typing import List
import os
import time
import requests
import yaml
from dotenv import load_dotenv
load_dotenv()

GENERATE_URL = os.getenv("MINER_SD_ENDPOINT")
CONFIG = yaml.load(open("sd_net/base_miner/config.yaml"), Loader=yaml.FullLoader)

def calculate_max_request_per_interval(stake: int):
    return CONFIG['blacklist']['tao_based_limitation']['max_requests_per_interval'] \
        * stake \
            // CONFIG['blacklist']['tao_based_limitation']['tao_base_level']


class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        self.validator_logs = {}
        
    def generate(self, prompt: str, seed: int, additional_params: dict) -> List[str]:
        data = {"prompt": prompt, "seed": seed, "additional_params": additional_params}

        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
        }

        response = requests.post(GENERATE_URL, headers=headers, json=data)
        images = response.json()['images']
        return images
    
    async def forward(
        self, synapse: sd_net.protocol.ImageGenerating
    ) -> sd_net.protocol.ImageGenerating:
        images = self.generate(synapse.prompt, synapse.seed, synapse.pipeline_params)
        synapse.images = images

        return synapse
    def check_limit(self, uid: str, stake: int):
        current_time = time.time()
        if uid not in self.validator_logs or (current_time - self.validator_logs[uid]['start_interval']) > CONFIG['tao_based_limitation']['interval']:
            self.validator_logs[uid] = {
                "start_interval": time.time(),
                "max_request": calculate_max_request_per_interval(stake=stake),
                "request_counter": 1,
            }
        else:
            self.validator_logs[uid]['request_counter'] += 1
            if self.validator_logs[uid]['request_counter'] > self.validator_logs[uid]['max_request']:
                self.validator_logs.pop(uid, None)
                return False
        return True
    async def blacklist(
        self, synapse: sd_net.protocol.ImageGenerating
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.Dummy): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """
        # TODO(developer): Define how miners should blacklist requests.
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        validator_uid = self.metagraph.hotkeys.index( synapse.dendrite.hotkey )
        stake = self.metagraph.stake[validator_uid].item()
        if stake < CONFIG['blacklist']['min_stake']:
            return True, "Validator doesn't have enough stake"
        if self.check_limit(uid=validator_uid, stake=stake):
            return True, "Limit exceeded"
        bt.logging.trace(
            f"Passed all blacklist checking!"
        )
        return False, "All passed!"


    async def priority(self, synapse: sd_net.protocol.ImageGenerating) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.Dummy): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        prirority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority
        )
        return prirority

# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(5)
