import bittensor as bt
from image_generation_subnet.protocol import Information
import torch
from image_generation_subnet.utils.volume_setting import (
    get_rate_limit_per_validator,
    MIN_RATE_LIMIT,
)

NO_OF_RECENT_SCORES = 10


class MinerInfo:
    def __init__(
        self, scores: list[float] = [], epoch_volume: int = 42, *args, **kwargs
    ):
        """
        TODO
        """
        self.scores: list[float] = scores
        self.epoch_volume: int = epoch_volume
        self.rate_limit = {}
        self.category: str = ""


class MinerManager:
    def __init__(self, validator):
        self.validator = validator
        self.all_uids = [int(uid.item()) for uid in self.validator.metagraph.uids]
        self.all_uids_info = {uid: MinerInfo() for uid in self.all_uids}

    def get_miner_info(self):
        """
        TODO
        """
        self.all_uids = [int(uid.item) for uid in self.validator.metagraph.uids]
        uid_to_axon = dict(zip(self.all_uids, self.validator.metagraph.axons))
        query_axons = [uid_to_axon[int(uid)] for uid in self.all_uids]
        synapse = Information()
        bt.logging.info("Requesting miner info using synapse Information")
        responses = self.validator.dendrite.query(
            query_axons,
            synapse,
            deserialize=False,
            timeout=10,
        )
        responses = {
            uid: response.response_dict
            for uid, response in zip(self.all_uids, responses)
        }
        responses = {k: v for k, v in responses.items() if v}
        return responses

    def update_miners_identity(self):
        """
        TODO
        """
        try:
            valid_miners_info = self.get_miner_info()
            if not valid_miners_info:
                bt.logging.warning(
                    "No active miner available. Skipping setting weights."
                )
            for uid, info in valid_miners_info.items():
                info = MinerInfo(**info)
                rate_limit_per_validator: dict = get_rate_limit_per_validator(
                    metagraph=self.validator.metagraph,
                    epoch_volume=info.epoch_volume,
                    log=False,
                )
                info.rate_limit = rate_limit_per_validator.get(
                    self.validator.uid, MIN_RATE_LIMIT
                )
                bt.logging.info(f"Rate limit for {uid}: {info.rate_limit}")

            bt.logging.success("Updated miner identity")
            return True
        except Exception:
            bt.logging.error("Update miner identity error!!!")
            return False

    def update_scores(self, uids, rewards):
        for uid, reward in zip(uids, rewards):
            self.all_uids_info[uid].scores.append(reward)
            self.all_uids_info[uid].scores = self.all_uids_info[uid]["scores"][
                -NO_OF_RECENT_SCORES:
            ]

    def get_on_chain_weights(self, category) -> torch.Tensor:
        weights = torch.zeros(len(self.all_uids))
        for uid, info in self.get_miner_uids(category):
            weights[int(uid)] = (
                sum(self.all_uids_info[uid].scores[-NO_OF_RECENT_SCORES:])
                / NO_OF_RECENT_SCORES
            )
        weights = weights + 1e-6
        weights = torch.clamp(weights, 0, 1)
        weights = weights / weights.sum()
        return weights
