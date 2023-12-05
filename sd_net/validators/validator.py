import time
import bittensor as bt
from sd_net.validators.utils.uids import get_random_uids
from sd_net.protocol import ImageGenerating
from sd_net.validators.reward import Rewarder
from template.base.validator import BaseValidatorNeuron


class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()
        # TODO(developer): Anything specific to your use case you can do here

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        prompt = "summer beach, blue sky"
        seed = 42
        miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
        responses = self.dendrite.query(
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse=ImageGenerating(prompt=prompt, seed=seed),
            deserialize=True,
        )
        bt.logging.info(f"Received responses: {responses}")
        raise
        #TODO: call api for verify & get reward
        # bt.logging.info(f"Scored responses: {rewards}")
        # self.update_scores(rewards, miner_uids)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Validator running...", time.time())
            time.sleep(5)
