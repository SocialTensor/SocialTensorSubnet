import time
from typing import Tuple
import bittensor as bt
from logicnet.base.miner import BaseMinerNeuron
import logicnet
from logicnet.protocol import LogicSynapse, Information
from logicnet.miner.forward import solve
import traceback
import openai

class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        self.validator_logs = {}
        self.volume_per_validator = (
            logicnet.utils.volume_setting.get_rate_limit_per_validator(
                self.metagraph,
                self.config.miner.total_volume,
                self.config.miner.min_stake,
            )
        )
        self.miner_info = {
            "epoch_volume": self.config.miner.total_volume,
            "category": "Logic",
        }
        self.num_processing_requests = 0
        self.total_request_in_interval = 0
        bt.logging.info(f"\033[1;32m🧠 Miner info: {self.miner_info}\033[0m")
        self.openai_client = openai.AsyncOpenAI(
            base_url=self.config.miner.llm_client.base_url,
            api_key=self.config.miner.llm_client.key,
        )

    async def forward(self, synapse: LogicSynapse) -> LogicSynapse:
        """
        Forward pass for the miner neuron. This function is called when a synapse is received by the miner neuron.
        By default, Miner will utilize the LLM API to solve the logic problem.
        """
        start_time = time.time()
        try:
            self.num_processing_requests += 1
            bt.logging.info(f"\033[1;33;44m🚀 Start processing request {self.num_processing_requests}\033[0m")
            synapse = await solve(
                synapse=synapse,
                openai_client=self.openai_client,
                model=self.config.miner.llm_client.model,
            )
            self.total_request_in_interval += 1
            
        except Exception as e:
            bt.logging.error(f"\033[1;31m❌ Error in forward: {e}\033[0m")
            traceback.print_exc()
    
        finally:
            process_time = time.time() - start_time
            bt.logging.info(f"\033[1;34;47m✅ Served request {self.num_processing_requests}: {round(process_time,2)} seconds\033[0m")
            
        return synapse

    async def forward_info(self, synapse: Information) -> Information:
        synapse.response_dict = self.miner_info
        return synapse

    async def blacklist_info(self, synapse: Information) -> Tuple[bool, str]:
        return False, "All passed!"

    async def blacklist(self, synapse: LogicSynapse) -> Tuple[bool, str]:
        bt.logging.info(f"synapse in blacklist {synapse}")
        try:
            if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
                # Ignore requests from unrecognized entities.
                bt.logging.trace(
                    f"\033[1;35m🛑 Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}\033[0m"
                )
                return True, "Unrecognized hotkey"

            validator_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
            stake = self.metagraph.stake[validator_uid].item()

            if validator_uid not in self.volume_per_validator:
                bt.logging.trace(
                    f"\033[1;35m🛑 Blacklisting {validator_uid}-validator has {stake} stake\033[0m"
                )
                return True, "Not enough stake"
            if logicnet.miner.check_limit(
                self,
                uid=validator_uid,
                stake=stake,
                volume_per_validator=self.volume_per_validator,
                interval=self.config.miner.limit_interval,
            ):
                bt.logging.trace(
                    f"\033[1;35m🛑 Blacklisting {validator_uid}-validator for exceeding the limit\033[0m"
                )
                return True, "Limit exceeded"

            return False, "All passed!"
        except Exception as e:
            bt.logging.error(f"\033[1;31m❌ Error in blacklist: {e}\033[0m")
            traceback.print_exc()
            return False, "All passed!"

    async def priority(self, synapse: LogicSynapse) -> float:
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"\033[1;36m🔝 Prioritizing {synapse.dendrite.hotkey} with value: {priority}\033[0m"
        )
        return priority

# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        start_time = time.time()
        while True:
            bt.logging.info(f"\033[1;32m⛏️ Miner running... {time.time()}\033[0m")
            if time.time() - start_time > 300:
                bt.logging.info(
                    f"\033[1;32m---Total request in last 5 minutes: {miner.total_request_in_interval}\033[0m"
                )
                start_time = time.time()
                miner.total_request_in_interval = 0
            try:
                miner.volume_per_validator = (
                    logicnet.utils.volume_setting.get_rate_limit_per_validator(
                        miner.metagraph,
                        miner.config.miner.total_volume,
                        miner.config.miner.min_stake,
                    )
                )
            except Exception as e:
                bt.logging.error(f"\033[1;31m❌ Error updating volume per validator: {e}\033[0m")
            time.sleep(60)
