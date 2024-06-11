import image_generation_subnet as ig_subnet
import asyncio

loop = asyncio.get_event_loop()


class Temp:
    class Config:
        generate_endpoint = "http://localhost:10006/generate"

    config = Config()


self = Temp()

synapse = ig_subnet.protocol.ImageGenerating(
    prompt="a cute cat",
    seed=42,
    pipeline_type="txt2img",
    pipeline_params={"style": "vivid", "size": "1024x1792"},
    timeout=64,
)

miner_response = loop.run_until_complete(
    ig_subnet.miner.forward.generate(self, synapse)
)

print(miner_response)


rewards = ig_subnet.validator.offline_reward.get_reward_dalle(
    synapse, [miner_response], [0]
)

print(rewards)
