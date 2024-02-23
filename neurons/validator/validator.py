import time
import bittensor as bt
import random
import torch
import image_generation_subnet.protocol as protocol
from image_generation_subnet.base.validator import BaseValidatorNeuron
from neurons.validator.validator_proxy import ValidatorProxy
from image_generation_subnet.validator import MinerManager
import image_generation_subnet as ig_subnet
import traceback


class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        self.miner_manager = MinerManager(self)

        bt.logging.info("load_state()")
        self.load_state()
        self.nicheimage_catalogue = {
            "TextToImage": {
                "base_synapse": protocol.TextToImage,
                "challenge_urls": [self.config.challenge.prompt],
                "models": {
                    "RealisticVision": {
                        "model_incentive_weight": 0.33,
                        "reward_url": self.config.reward.text_to_image.RealisticVision,
                        "inference_params": {
                            "num_inference_steps": 30,
                            "negative_prompt": "out of frame, nude, duplicate, watermark, signature, mutated, text, blurry, worst quality, low quality, artificial, texture artifacts, jpeg artifacts",
                        },
                        "timeout": 4,
                    },
                    "DreamShaperXL": {
                        "model_incentive_weight": 0.33,
                        "reward_url": self.config.reward.text_to_image.DreamShaperXL,
                        "inference_params": {
                            "num_inference_steps": 6,
                            "width": 1024,
                            "height": 1024,
                            "guidance_scale": 2,
                            "negative_prompt": "out of frame, nude, duplicate, watermark, signature, mutated, text, blurry, worst quality, low quality, artificial, texture artifacts, jpeg artifacts",
                        },
                        "timeout": 4,
                    },
                    "AnimeV3": {
                        "model_incentive_weight": 0.34,
                        "reward_url": self.config.reward.text_to_image.AnimeV3,
                        "inference_params": {
                            "num_inference_steps": 20,
                            "width": 1024,
                            "height": 1024,
                            "guidance_scale": 7.0,
                            "negative_prompt": "out of frame, nude, duplicate, watermark, signature, mutated, text, blurry, worst quality, low quality, artificial, texture artifacts, jpeg artifacts",
                        },
                        "timeout": 12,
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
                    "DreamShaperXL": {
                        "model_incentive_weight": 1.0,
                        "reward_url": self.config.reward.image_to_image.DreamShaperXL,
                        "inference_params": {
                            "num_inference_steps": 6,
                            "guidance_scale": 2,
                            "strength": 0.9,
                            "negative_prompt": "Compression artifacts, bad art, worst quality, low quality, plastic, fake, bad limbs, conjoined, featureless, bad features, incorrect objects, watermark, signature, logo",
                        },
                        "timeout": 4,
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
                        "model_incentive_weight": 1.0,
                        "reward_url": self.config.reward.controlnet_text_to_image.DreamShaper,
                        "inference_params": {
                            "num_inference_steps": 30,
                            "guidance_scale": 7.0,
                            "negative_prompt": "worst quality, greyscale, low quality, bad art, plastic, fake, bad limbs, conjoined, featureless, bad features, incorrect objects, watermark, signature, logo",
                        },
                        "timeout": 4,
                    },
                },
                "category_incentive_weight": 0.33,
            },
        }
        self.max_validate_batch = 5
        if self.config.proxy.port:
            try:
                self.validator_proxy = ValidatorProxy(self)
                bt.logging.info("Validator proxy started succesfully")
            except Exception:
                bt.logging.warning(
                    "Warning, proxy did not start correctly, so no one can query through your validator. Error message: "
                    + traceback.format_exc()
                )
        self.miner_manager.update_miners_identity()

    def forward(self):
        """
        Validator forward pass. Consists of:
        - Querying all miners to get their model_name and category
        - Looping through all the categories and models
        - Generating the challenge: prompt and image[optional]
        - Batching miners and sending the challenge
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """

        bt.logging.info("Updating available models & uids")
        self.miner_manager.update_miners_identity()

        for category in self.nicheimage_catalogue.keys():
            for model_name in self.nicheimage_catalogue[category]["models"].keys():
                reward_url = self.nicheimage_catalogue[category]["models"][model_name][
                    "reward_url"
                ]
                available_uids = self.miner_manager.get_miner_uids(model_name, category)
                if not available_uids:
                    bt.logging.warning(
                        f"No active miner available for specified model {category}-{model_name}. Skipping setting weights."
                    )
                    continue

                bt.logging.info(
                    f"Available uids for {category}-{model_name}: {available_uids}"
                )

                synapses, batched_uids = self.prepare_challenge(
                    available_uids, category, model_name
                )

                for synapse, uids in zip(synapses, batched_uids):
                    try:
                        base_synapse = synapse.copy()
                        responses = self.dendrite.query(
                            axons=[self.metagraph.axons[int(uid)] for uid in uids],
                            synapse=synapse,
                            deserialize=False,
                        )

                        bt.logging.info("Received responses, calculating rewards")
                        uids, rewards = ig_subnet.validator.get_reward(
                            reward_url, base_synapse, responses, uids
                        )

                        bt.logging.info(f"Scored responses: {rewards}")

                        self.miner_manager.update_scores(uids, rewards)
                    except Exception as e:
                        bt.logging.error(
                            f"Error while processing forward pass for {category}-{model_name}: {e}"
                        )
                        continue

        self.update_scores_on_chain()
        self.save_state()

    def prepare_challenge(self, available_uids, category, model_name):
        batch_size = random.randint(1, 5)
        random.shuffle(available_uids)
        batched_uids = [
            available_uids[i * batch_size : (i + 1) * batch_size]
            for i in range(len(available_uids) // batch_size)
        ]
        num_batch = len(batched_uids)
        synapses = [
            self.nicheimage_catalogue[category]["base_synapse"]()
            for _ in range(num_batch)
        ]
        for synapse in synapses:
            synapse.pipeline_params.update(
                self.nicheimage_catalogue[category]["models"][model_name][
                    "inference_params"
                ]
            )
            synapse.seed = random.randint(0, 1e9)
        for challenge_url in self.nicheimage_catalogue[category]["challenge_urls"]:
            synapses = ig_subnet.validator.get_challenge(challenge_url, synapses)
        return synapses, batched_uids

    def update_scores_on_chain(self):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        weights = torch.zeros(len(self.miner_manager.all_uids))
        for category in self.nicheimage_catalogue.keys():
            for model_name in self.nicheimage_catalogue[category]["models"].keys():
                model_specific_weights = self.miner_manager.get_model_specific_weights(
                    model_name, category
                )
                model_specific_weights = (
                    model_specific_weights
                    * self.nicheimage_catalogue[category]["models"][model_name][
                        "model_incentive_weight"
                    ]
                    * self.nicheimage_catalogue[category]["category_incentive_weight"]
                )
                bt.logging.info(
                    f"model_specific_weights for {category}-{model_name}\n{model_specific_weights}"
                )
                weights = weights + model_specific_weights

        # Check if rewards contains NaN values.
        if torch.isnan(weights).any():
            bt.logging.warning(f"NaN values detected in weights: {weights}")
            # Replace any NaN values in rewards with 0.
            weights = torch.nan_to_num(weights, 0)
        self.scores: torch.FloatTensor = weights
        bt.logging.success(f"Updated scores: {self.scores}")

    def save_state(self):
        """Saves the state of the validator to a file."""

        torch.save(
            {
                "step": self.step,
                "all_uids_info": self.miner_manager.all_uids_info,
            },
            self.config.neuron.full_path + "/state.pt",
        )

    def load_state(self):
        """Loads the state of the validator from a file."""

        # Load the state of the validator from file.
        try:
            path = self.config.neuron.full_path + "/state.pt"
            bt.logging.info("Loading validator state from: " + path)
            state = torch.load(path)
            self.step = state["step"]
            self.miner_manager.all_uids_info = state["all_uids_info"]
            bt.logging.info("Succesfully loaded state")
        except Exception as e:
            self.step = 0
            bt.logging.info("Could not find previously saved state.", e)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Validator running...", time.time())
            time.sleep(60)
