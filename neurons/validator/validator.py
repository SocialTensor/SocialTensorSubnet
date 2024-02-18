import time
import bittensor as bt
import random
import torch
import image_generation_subnet.protocol as protocol
from image_generation_subnet.base.validator import BaseValidatorNeuron
from neurons.validator.validator_proxy import ValidatorProxy
import image_generation_subnet as ig_subnet
import traceback


class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        self.all_uids = [int(uid.item()) for uid in self.metagraph.uids]

        bt.logging.info("load_state()")
        self.load_state()
        self.category_models = {
            "TextToImage": {
                "base_synapse": protocol.TextToImage,
                "challenge_urls": [self.config.challenge.prompt],
                "models": {
                    "RealisticVision": {
                        "model_incentive_weight": 0.33,
                        "reward_url": self.config.reward.text_to_image.RealisticVision,
                        "inference_params": {"num_inference_steps": 30},
                        "timeout": 12,
                    },
                    "SDXLTurbo": {
                        "model_incentive_weight": 0.33,
                        "reward_url": self.config.reward.text_to_image.SDXLTurbo,
                        "inference_params": {
                            "num_inference_steps": 4,
                            "width": 512,
                            "height": 512,
                            "guidance_scale": 0.5,
                        },
                        "timeout": 4,
                    },
                    "AnimeV3": {
                        "model_incentive_weight": 0.34,
                        "reward_url": self.config.reward.text_to_image.AnimeV3,
                        "inference_params": {
                            "prompt_template": "anime key visual, acrylic painting, %s, pixiv fanbox, natural lighting",
                            "num_inference_steps": 20,
                            "width": 576,
                            "height": 960,
                            "guidance_scale": 7.0,
                            "negative_prompt": "(out of frame), nude, duplicate, watermark, signature, mutated, text, blurry, worst quality, low quality, artificial, texture artifacts, jpeg artifacts",
                        },
                        "timeout": 20,
                    },
                },
                "category_incentive_weight": 0.34,
            },
            "ImageToImage": {
                "base_synapse": protocol.ImageToImage,
                "challenge_urls": [
                    self.config.challenge.prompt,
                    self.config.challenge.image,
                ],
                "models": {
                    "DreamShaper": {
                        "model_incentive_weight": 1.0,
                        "reward_url": self.config.reward.image_to_image.DreamShaper,
                        "inference_params": {
                            "num_inference_steps": 30,
                            "guidance_scale": 7.0,
                            "negative_prompt": "Compression artifacts, bad art, worst quality, low quality, plastic, fake, bad limbs, conjoined, featureless, bad features, incorrect objects, watermark, signature, logo",
                        },
                        "timeout": 20,
                    },
                },
                "category_incentive_weight": 0.33,
            },
            "ControlNetTextToImage": {
                "base_synapse": protocol.ControlNetTextToImage,
                "challenge_urls": [
                    self.config.challenge.prompt,
                    self.config.challenge.image,
                ],
                "models": {
                    "DreamShaper": {
                        "model_incentive_weight": 0.5,
                        "reward_url": self.config.reward.controlnet_text_to_image.DreamShaper,
                        "inference_params": {
                            "num_inference_steps": 30,
                            "guidance_scale": 7.0,
                            "negative_prompt": "worst quality, greyscale, low quality, bad art, plastic, fake, bad limbs, conjoined, featureless, bad features, incorrect objects, watermark, signature, logo",
                        },
                        "timeout": 8,
                    },
                },
                "category_incentive_weight": 0.33,
            },
        }
        self.max_validate_batch = 5

        self.update_active_models_func = ig_subnet.validator.update_active_models

        if self.config.proxy.port:
            try:
                self.validator_proxy = ValidatorProxy(self)
                bt.logging.info("Validator proxy started succesfully")
            except Exception as e:
                bt.logging.warning(
                    "Warning, proxy did not start correctly, so no one can query through your validator. Error message: "
                    + traceback.format_exc()
                )
        self.update_active_models_func(self)

    def forward(self):
        """
        Validator forward pass. Consists of:
        - Querying all miners to get what model they run
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """

        bt.logging.info("Updating available models & uids")
        self.update_active_models_func(self)

        for category in self.category_models.keys():
            category_uids = [
                uid
                for uid in self.all_uids_info.keys()
                if self.all_uids_info[uid]["category"] == category
            ]
            if not category_uids:
                bt.logging.warning(
                    f"No active miner available for specified category {category}. Skipping setting weights."
                )
                continue

            bt.logging.info(f"Available uids for {category}: {category_uids}")

            for model_name in self.category_models[category]["models"].keys():
                challenge_urls = self.category_models[category]["challenge_urls"]
                reward_url = self.category_models[category]["models"][model_name][
                    "reward_url"
                ]
                model_uids = [
                    uid
                    for uid in category_uids
                    if self.all_uids_info[uid]["model_name"] == model_name
                ]
                if not model_uids:
                    bt.logging.warning(
                        f"No active miner available for specified model {model_name}. Skipping setting weights."
                    )
                    continue

                bt.logging.info(f"Available uids for {model_name}: {model_uids}")

                num_batch = (
                    len(model_uids) + self.max_validate_batch - 1
                ) // self.max_validate_batch

                seeds = [random.randint(0, 1e9) for _ in range(num_batch)]
                batched_uids = [
                    model_uids[
                        i * self.max_validate_batch : (i + 1) * self.max_validate_batch
                    ]
                    for i in range(num_batch)
                ]

                synapses = [
                    self.category_models[category]["base_synapse"]()
                    for _ in range(num_batch)
                ]
                for i, synapse in enumerate(synapses):
                    synapse.pipeline_params.update(
                        self.category_models[category]["models"][model_name][
                            "inference_params"
                        ]
                    )
                    synapse.seed = seeds[i]
                for challenge_url in challenge_urls:
                    synapses = ig_subnet.validator.get_challenge(
                        challenge_url, synapses
                    )

                for synapse, uids in zip(synapses, batched_uids):
                    base_synapse = synapse.copy()
                    responses = self.dendrite.query(
                        axons=[self.metagraph.axons[int(uid)] for uid in uids],
                        synapse=synapse,
                        deserialize=False,
                    )
                    valid_uids = [
                        uid
                        for uid, response in zip(uids, responses)
                        if response.is_success
                    ]
                    invalid_uids = [
                        uid
                        for uid, response in zip(uids, responses)
                        if not response.is_success
                    ]
                    responses = [
                        response for response in responses if response.is_success
                    ]
                    process_times = [
                        response.dendrite.process_time for response in responses
                    ]

                    bt.logging.info("Received responses, calculating rewards")
                    rewards = ig_subnet.validator.get_reward(
                        reward_url, base_synapse, responses
                    )
                    rewards = ig_subnet.validator.add_time_penalty(
                        rewards, process_times
                    )
                    rewards = [round(num, 3) for num in rewards]

                    total_uids = valid_uids + invalid_uids
                    rewards = rewards + [0] * len(invalid_uids)

                    bt.logging.info(f"Scored responses: {rewards}")

                    for i in range(len(total_uids)):
                        uid = total_uids[i]
                        self.all_uids_info[uid]["scores"].append(rewards[i])
                        self.all_uids_info[uid]["scores"] = self.all_uids_info[uid][
                            "scores"
                        ][-10:]

        self.update_scores_on_chain()
        self.save_state()

    def update_scores_on_chain(self):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        weights = torch.zeros(len(self.all_uids))
        for category in self.category_models.keys():
            for model_name in self.category_models[category]["models"].keys():
                model_specific_weights = torch.zeros(len(self.all_uids))

                for uid in self.all_uids_info.keys():
                    if (
                        self.all_uids_info[uid]["model_name"] == model_name
                        and self.all_uids_info[uid]["category"] == category
                    ):
                        num_past_to_check = 5
                        model_specific_weights[int(uid)] = (
                            sum(self.all_uids_info[uid]["scores"][-num_past_to_check:])
                            / num_past_to_check
                        )
                model_specific_weights = torch.clamp(model_specific_weights, 0, 1)
                tensor_sum = torch.sum(model_specific_weights)
                # Normalizing the tensor
                if tensor_sum > 0:
                    model_specific_weights = model_specific_weights / tensor_sum
                else:
                    continue
                # Correcting reward
                model_specific_weights = (
                    model_specific_weights
                    * self.category_models[category]["models"][model_name][
                        "model_incentive_weight"
                    ]
                    * self.category_models[category]["category_incentive_weight"]
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


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Validator running...", time.time())
            time.sleep(60)
