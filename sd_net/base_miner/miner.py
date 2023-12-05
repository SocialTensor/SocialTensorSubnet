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


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(5)
