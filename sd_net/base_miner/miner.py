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
    return CONFIG["blacklist"]["tao_based_limitation"]["max_requests_per_interval"] * (
        stake // CONFIG["blacklist"]["tao_based_limitation"]["tao_base_level"]
    )


class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        self.validator_logs = {}

    def generate(self, prompt: str, seed: int, additional_params: dict) -> List[str]:
        data = {"prompt": prompt, "seed": seed, "additional_params": additional_params}

        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }

        response = requests.post(GENERATE_URL, headers=headers, json=data)
        images = response.json()["images"]
        return images

    async def forward(
        self, synapse: sd_net.protocol.ImageGenerating
    ) -> sd_net.protocol.ImageGenerating:
        print(synapse)
        images = self.generate(synapse.prompt, synapse.seed, synapse.pipeline_params)
        synapse.images = images

        return synapse

    def check_limit(self, uid: str, stake: int):
        current_time = time.time()
        print(self.validator_logs)
        if uid not in self.validator_logs:
            self.validator_logs[uid] = {
                "start_interval": time.time(),
                "max_request": calculate_max_request_per_interval(stake=stake),
                "request_counter": 1,
            }
        elif (
            time.time() - self.validator_logs[uid]["start_interval"]
            > CONFIG["blacklist"]["tao_based_limitation"]["interval"]
        ):
            self.validator_logs[uid] = {
                "start_interval": time.time(),
                "max_request": calculate_max_request_per_interval(stake=stake),
                "request_counter": 1,
            }
            print(f"RESET INTERVAL OF {uid}")
        else:
            self.validator_logs[uid]["request_counter"] += 1
            if (
                self.validator_logs[uid]["request_counter"]
                > self.validator_logs[uid]["max_request"]
            ):
                return True
        return False

    async def blacklist(
        self, synapse: sd_net.protocol.ImageGenerating
    ) -> typing.Tuple[bool, str]:
        # # TODO(developer): Define how miners should blacklist requests.
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        validator_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        stake = self.metagraph.stake[validator_uid].item()
        if stake < CONFIG["blacklist"]["min_stake"]:
            bt.logging.trace(
                f"Blacklisting {validator_uid}-validator has {stake} stake"
            )
            return True, "Validator doesn't have enough stake"
        if self.check_limit(uid=validator_uid, stake=stake):
            return True, "Limit exceeded"

        return False, "All passed!"

    async def priority(self, synapse: sd_net.protocol.ImageGenerating) -> float:
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
