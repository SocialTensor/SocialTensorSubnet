import time
import bittensor as bt
import random
import torch
from image_generation_subnet.protocol import ImageGenerating
from image_generation_subnet.base.validator import BaseValidatorNeuron
from neurons.validator.validator_proxy import ValidatorProxy
import image_generation_subnet as ig_subnet
from traceback import print_exception


class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()
        self.all_uids = [int(uid) for uid in self.metagraph.uids]
        self.supporting_models = {
            "RealisticVision": {
                "uids": [],
                "incentive_weight": 1.0,
                "checking_endpoint": "http://127.0.0.1:10002/verify",
            }
        }
        self.validator_proxy = ValidatorProxy(
            self.uid,
            self.metagraph,
            self.dendrite,
            self.config.proxy.port,
            self.config.proxy.market_registering_url,
            self.supporting_models,
            self.config.proxy.public_ip,
            self.scores,
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
        prompt = ig_subnet.validator.get_prompt(
            self, seed=seed, prompt_url=self.config.prompt_generating_endpoint
        )
        model_name = random.choice(list(self.supporting_models.keys()))

        bt.logging.info(f"Received request for {model_name} model")
        bt.logging.info("Updating available models & uids")
        ig_subnet.validator.update_active_models(self)

        available_uids = self.supporting_models[model_name]["uids"]

        if not available_uids:
            bt.logging.warning(
                "No active miner available for specified model. Skipping setting weights."
            )
            return
        else:
            bt.logging.info(f"Available uids: {available_uids}")

        synapse = ImageGenerating(
            prompt=prompt,
            seed=seed,
            pipeline_params={"num_inference_steps": steps},
            model_name=model_name,
        )
        responses = self.dendrite.query(
            axons=[self.metagraph.axons[uid] for uid in available_uids],
            synapse=synapse,
            deserialize=False,
        )

        bt.logging.info("Received responses, calculating rewards")
        rewards = ig_subnet.validator.get_reward(
            self, self.config.reward_endpoint, responses, synapse
        )
        rewards = torch.FloatTensor(rewards)
        rewards = rewards * self.supporting_models[model_name]["incentive_weight"]
        bt.logging.info(f"Scored responses: {rewards}")
        self.update_scores(rewards, available_uids)
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
        self.resync_metagraph()

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

                # Update the validator proxy
                self.validator_proxy.metagraph = self.metagraph
                self.validator_proxy.supporting_models = self.supporting_models
                self.validator_proxy.scores = self.scores

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


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Validator running...", time.time())
            time.sleep(60)
