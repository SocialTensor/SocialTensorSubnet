import time
from typing import Tuple, TypeVar
import bittensor as bt
from image_generation_subnet.base.miner import BaseMinerNeuron
import image_generation_subnet
from image_generation_subnet.protocol import (
    TextToImage,
    NicheImageProtocol,
    ImageToImage,
    ControlNetTextToImage,
)

T = TypeVar("T", bound=bt.Synapse)


class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        self.validator_logs = {}
        self.miner_info = image_generation_subnet.miner.set_info(self)
        self.axon.attach(
            forward_fn=self.forward_info,
            blacklist_fn=self.blacklist_info,
        ).attach(
            forward_fn=self.forward_text_to_image,
            blacklist_fn=self.blacklist_text_to_image,
        ).attach(
            forward_fn=self.forward_controlnet_text_to_image,
            blacklist_fn=self.blacklist_controlnet_text_to_image,
        ).attach(
            forward_fn=self.forward_image_to_image,
            blacklist_fn=self.blacklist_image_to_image,
        )

    async def forward(self, synapse: bt.Synapse) -> T:
        bt.logging.info(f"synapse prompt: {synapse.prompt}")
        if synapse.request_dict:
            synapse.response_dict = self.miner_info
            bt.logging.info(f"Response dict: {self.miner_info}")
        else:
            synapse = image_generation_subnet.miner.generate(self, synapse)
        return synapse

    async def forward_info(self, synapse: NicheImageProtocol) -> NicheImageProtocol:
        return await self.forward(synapse)

    async def forward_text_to_image(self, synapse: TextToImage) -> TextToImage:
        return await self.forward(synapse)

    async def forward_controlnet_text_to_image(
        self, synapse: ControlNetTextToImage
    ) -> ControlNetTextToImage:
        return await self.forward(synapse)

    async def forward_image_to_image(self, synapse: ImageToImage) -> ImageToImage:
        return await self.forward(synapse)

    async def blacklist(self, synapse: bt.Synapse) -> Tuple[bool, str]:
        bt.logging.info(f"synapse in blacklist {synapse}")

        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        validator_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        stake = self.metagraph.stake[validator_uid].item()

        if image_generation_subnet.miner.check_min_stake(stake, validator_uid):
            bt.logging.trace(
                f"Blacklisting {validator_uid}-validator has {stake} stake"
            )
            return True, "Not enough stake"
        if image_generation_subnet.miner.check_limit(
            self, uid=validator_uid, stake=stake
        ):
            bt.logging.trace(
                f"Blacklisting {validator_uid}-validator for exceeding the limit"
            )
            return True, "Limit exceeded"

        return False, "All passed!"

    async def blacklist_info(self, synapse: NicheImageProtocol) -> Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def blacklist_text_to_image(self, synapse: TextToImage) -> Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def blacklist_controlnet_text_to_image(
        self, synapse: ControlNetTextToImage
    ) -> Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def blacklist_image_to_image(self, synapse: ImageToImage) -> Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority(self, synapse: NicheImageProtocol) -> float:
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

    async def priority_text_to_image(self, synapse: TextToImage) -> float:
        return await self.priority(synapse)

    async def priority_controlnet_text_to_image(
        self, synapse: ControlNetTextToImage
    ) -> float:
        return await self.priority(synapse)

    async def priority_image_to_image(self, synapse: ImageToImage) -> float:
        return await self.priority(synapse)


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(5)
