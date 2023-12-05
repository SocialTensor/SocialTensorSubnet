import time
import typing
import bittensor as bt
import template
from template.base.miner import BaseMinerNeuron
from diffusers import StableDiffusionXLPipeline
import sd_net
import torch
from sd_net.protocol import pil_image_to_base64


class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            "models/unstable_sdxl.safetensors"
        )
        self.pipe.to("cuda")

    async def forward(
        self, synapse: sd_net.protocol.ImageGenerating
    ) -> sd_net.protocol.ImageGenerating:
        images = self.pipe(
            synapse.prompt,
            generator=torch.Generator(device=self.pipe.device).manual_seed(
                synapse.seed
            ),
            **synapse.pipeline_params,
        ).images
        images = [pil_image_to_base64(image) for image in images]
        synapse.images = images

        return synapse
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
        # if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
        #     # Ignore requests from unrecognized entities.
        #     bt.logging.trace(
        #         f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
        #     )
        #     return True, "Unrecognized hotkey"

        # bt.logging.trace(
        #     f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        # )
        # return False, "Hotkey recognized!"
        return False, "Hotkey recognized!"


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
