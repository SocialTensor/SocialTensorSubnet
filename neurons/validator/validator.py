import time
import bittensor as bt
import random
import torch
from image_generation_subnet.protocol import ImageGenerating
from image_generation_subnet.base.validator import BaseValidatorNeuron
from neurons.validator.validator_proxy import ValidatorProxy
from image_generation_subnet.validator import MinerManager
import image_generation_subnet as ig_subnet
import traceback
import yaml
import threading

MODEL_CONFIGS = yaml.load(
    open("generation_models/configs/model_config.yaml"), yaml.FullLoader
)


class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.challenge_urls = {
            "txt2img": [self.config.challenge.prompt],
            "img2img": [self.config.challenge.prompt, self.config.challenge.image],
            "controlnet_txt2img": [
                self.config.challenge.prompt,
                self.config.challenge.image,
            ],
            "gojourney": [
                self.config.challenge.prompt,
                ig_subnet.validator.get_promptGoJouney,
            ],
        }
        # TODO: Balancing Incentive Weights
        self.nicheimage_catalogue = {
            "GoJourney": {
                "model_incentive_weight": 0.04,
                "supporting_pipelines": MODEL_CONFIGS["GoJourney"]["params"][
                    "supporting_pipelines"
                ],
                "reward_url": ig_subnet.validator.get_reward_GoJourney,
                "timeout": 12,
                "inference_params": {},
            },
            "DreamShaper": {
                "model_incentive_weight": 0.06,
                "supporting_pipelines": MODEL_CONFIGS["DreamShaper"]["params"][
                    "supporting_pipelines"
                ],
                "reward_url": self.config.reward_url.DreamShaper,
                "inference_params": {
                    "num_inference_steps": 30,
                    "width": 512,
                    "height": 768,
                    "guidance_scale": 7,
                    "negative_prompt": "out of frame, nude, duplicate, watermark, signature, mutated, text, blurry, worst quality, low quality, artificial, texture artifacts, jpeg artifacts",
                },
                "timeout": 12,
            },
            "RealisticVision": {
                "supporting_pipelines": MODEL_CONFIGS["RealisticVision"]["params"][
                    "supporting_pipelines"
                ],
                "model_incentive_weight": 0.26,
                "reward_url": self.config.reward_url.RealisticVision,
                "inference_params": {
                    "num_inference_steps": 30,
                    "negative_prompt": "out of frame, nude, duplicate, watermark, signature, mutated, text, blurry, worst quality, low quality, artificial, texture artifacts, jpeg artifacts",
                },
                "timeout": 12,
            },
            "RealitiesEdgeXL": {
                "supporting_pipelines": MODEL_CONFIGS["RealitiesEdgeXL"]["params"][
                    "supporting_pipelines"
                ],
                "model_incentive_weight": 0.30,
                "reward_url": self.config.reward_url.RealitiesEdgeXL,
                "inference_params": {
                    "num_inference_steps": 7,
                    "width": 1024,
                    "height": 1024,
                    "guidance_scale": 5.5,
                },
                "timeout": 12,
            },
            "AnimeV3": {
                "supporting_pipelines": MODEL_CONFIGS["AnimeV3"]["params"][
                    "supporting_pipelines"
                ],
                "model_incentive_weight": 0.34,
                "reward_url": self.config.reward_url.AnimeV3,
                "inference_params": {
                    "num_inference_steps": 25,
                    "width": 1024,
                    "height": 1024,
                    "guidance_scale": 7.0,
                    "negative_prompt": "out of frame, nude, duplicate, watermark, signature, mutated, text, blurry, worst quality, low quality, artificial, texture artifacts, jpeg artifacts",
                },
                "timeout": 12,
            },
        }
        self.max_validate_batch = 5
        self.miner_manager = MinerManager(self)
        self.load_state()
        try:
            self.validator_proxy = ValidatorProxy(self)
            bt.logging.info("Validator proxy started succesfully")
        except Exception:
            if self.config.proxy.port:
                bt.logging.warning(
                    "Warning, proxy did not start correctly, so no one can query through your validator. Error message: "
                    + traceback.format_exc()
                )
            else:
                bt.logging.warning("Share validator info to owner failed")
        self.miner_manager.update_miners_identity()
        self.flattened_uids = []

    def forward(self):
        """
        Validator forward pass. Consists of:
        - Querying all miners to get their model_name
        - Forwarding requests to miners
        - Calculating rewards based on responses
        - Updating scores based on rewards
        - Saving the state
        """

        bt.logging.info("Updating available models & uids")
        num_concurrent_forward = self.config.num_concurrent_forward
        self.miner_manager.update_miners_identity()
        _uids = self.miner_manager.all_uids
        rate_limit_per_uid = [
            self.miner_manager.all_uids_info[uid]["rate_limit"] for uid in _uids
        ]
        for uid, rate_limit in zip(_uids, rate_limit_per_uid):
            self.flattened_uids += [uid] * rate_limit
        random.shuffle(self.flattened_uids)
        _model_names = [
            self.miner_manager.all_uids_info[uid]["model_name"] for uid in _uids
        ]
        model_names = []
        uids = []

        for uid, model_name in zip(_uids, _model_names):
            if model_name in self.nicheimage_catalogue:
                model_names.append(model_name)
                uids.append(uid)

        maximum_loop_time = sum(
            [
                self.nicheimage_catalogue[model_name]["timeout"]
                for model_name in model_names
            ]
        )
        loop_base_time = self.config.loop_base_time  # default is 600 seconds
        forward_batch_size = len(self.flattened_uids) // num_concurrent_forward
        sleep_per_batch = loop_base_time / num_concurrent_forward
        threads = []
        bt.logging.info(
            (
                f"Forwarding {len(self.flattened_uids)} uids\n"
                f"Use {num_concurrent_forward} concurrent forward passes\n"
                f"Each forward pass will forward {forward_batch_size} uids\n"
                f"Sleeping {sleep_per_batch} seconds per batch to ensure the loop time is around {loop_base_time} seconds\n"
                f"Total loop time should be around {maximum_loop_time} seconds\n"
            )
        )
        while self.flattened_uids:
            batch_uids = self.flattened_uids[:forward_batch_size]
            batch_model_names = model_names[:forward_batch_size]
            pipeline_types = [
                random.choice(
                    self.nicheimage_catalogue[model_name]["supporting_pipelines"]
                )
                for model_name in batch_model_names
            ]

            del self.flattened_uids[:forward_batch_size]
            thread = threading.Thread(
                target=self.async_query_and_reward,
                args=(batch_uids, batch_model_names, pipeline_types),
            )
            threads.append(thread)
            thread.start()
            time.sleep(sleep_per_batch)
        for thread in threads:
            thread.join()
        self.update_scores_on_chain()
        self.save_state()

    def async_query_and_reward(
        self, uids: list[int], model_names: list[str], pipeline_types: list[str]
    ):
        batch_by_model_pipeline = {}
        for uid, model_name, pipeline_type in zip(uids, model_names, pipeline_types):
            batch_by_model_pipeline.setdefault((model_name, pipeline_type), []).append(
                uid
            )
        for (model_name, pipeline_type), uids in batch_by_model_pipeline.items():
            try:
                reward_url = self.nicheimage_catalogue[model_name]["reward_url"]
                synapses, batched_uids = self.prepare_challenge(
                    uids, model_name, pipeline_type
                )
                for synapse, uids in zip(synapses, batched_uids):
                    if not synapse:
                        continue
                    base_synapse = synapse.copy()
                    responses = self.dendrite.query(
                        axons=[self.metagraph.axons[int(uid)] for uid in uids],
                        synapse=synapse,
                        deserialize=False,
                        timeout=self.nicheimage_catalogue[model_name]["timeout"],
                    )

                    bt.logging.info("Received responses, calculating rewards")
                    if callable(reward_url):
                        uids, rewards = reward_url(base_synapse, responses, uids)
                    else:
                        uids, rewards = ig_subnet.validator.get_reward(
                            reward_url, base_synapse, responses, uids
                        )

                    bt.logging.info(f"Scored responses: {rewards}")

                    self.miner_manager.update_scores(uids, rewards)
            except Exception as e:
                bt.logging.error(
                    f"Error while processing forward pass for {model_name}: {e}"
                )
                bt.logging.error(traceback.print_exc())
                continue

    def prepare_challenge(self, available_uids, model_name, pipeline_type):
        batch_size = random.randint(1, 5)
        random.shuffle(available_uids)
        batched_uids = [
            available_uids[i * batch_size : (i + 1) * batch_size]
            for i in range((len(available_uids) + batch_size - 1) // batch_size)
        ]
        num_batch = len(batched_uids)
        synapses = [
            ImageGenerating(pipeline_type=pipeline_type, model_name=model_name)
            for _ in range(num_batch)
        ]
        for synapse in synapses:
            synapse.pipeline_params.update(
                self.nicheimage_catalogue[model_name]["inference_params"]
            )
            synapse.seed = random.randint(0, 1e9)
        for challenge_url in self.challenge_urls[pipeline_type]:
            if callable(challenge_url):
                synapses = challenge_url(synapses)
            else:
                assert isinstance(challenge_url, str)
                synapses = ig_subnet.validator.get_challenge(challenge_url, synapses)
        return synapses, batched_uids

    def update_scores_on_chain(self):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        weights = torch.zeros(len(self.miner_manager.all_uids))
        for model_name in self.nicheimage_catalogue.keys():
            model_specific_weights = self.miner_manager.get_model_specific_weights(
                model_name
            )
            model_specific_weights = (
                model_specific_weights
                * self.nicheimage_catalogue[model_name]["model_incentive_weight"]
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
