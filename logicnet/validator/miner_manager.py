import bittensor as bt
from logicnet.protocol import Information
import torch
from logicnet.utils.volume_setting import (
    get_rate_limit_per_validator,
    MIN_RATE_LIMIT,
)
import traceback

NO_OF_RECENT_SCORES = 10


class MinerInfo:
    def __init__(
        self,
        category: str = "",
        scores: list[float] = [],
        epoch_volume: int = 42,
        reward_scale: float = 0.0,
        *args,
        **kwargs,
    ):
        """Miner Infomation to be refreshed every epoch

        Args:
            category (str, optional): Category of running miner. Defaults to "" if this uid is inactive.
            scores (list[float], optional): Some recent scores of miner. Defaults to [] if this uid is inactive.
            epoch_volume (int, optional): No of requests / epoch commited by miner. Defaults to 42.
            reward_scale (float, optional): The scale value applied to miner reward each epoch. Defaults to 0.0.
        """
        self.scores: list[float] = scores
        self.epoch_volume: int = epoch_volume
        self.rate_limit = 0
        self.category: str = category
        self.reward_scale: float = reward_scale

    def __str__(self):
        return f"MinerInfo: {self.category} {self.scores} {self.epoch_volume} {self.rate_limit} {self.reward_scale}"

    def __repr__(self):
        return f"MinerInfo: {self.category} {self.scores} {self.epoch_volume} {self.rate_limit} {self.reward_scale}"

    def to_dict(self):
        return {
            "category": self.category,
            "scores": self.scores,
            "epoch_volume": self.epoch_volume,
            "rate_limit": self.rate_limit,
            "reward_scale": self.reward_scale,
        }


class MinerManager:
    def __init__(self, validator):
        self.validator = validator
        self.all_uids = [int(uid.item()) for uid in self.validator.metagraph.uids]
        self.all_uids_info = {uid: MinerInfo() for uid in self.all_uids}

    def get_miner_info(self):
        """
        QUERY MINER's INFORMATION SYNAPSE
        """
        self.all_uids = [int(uid.item()) for uid in self.validator.metagraph.uids]
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
        Update miner's identity with new information
        VALIDATOR calculates the rate limit and reward scale for each miner based on the epoch volume
        """
        try:
            valid_miners_info = self.get_miner_info()
            if not valid_miners_info:
                bt.logging.warning(
                    "No active miner available. Skipping setting weights."
                )
            miner_distribution = {}
            for uid, info in valid_miners_info.items():
                info = MinerInfo(**info)
                rate_limit_per_validator: dict = get_rate_limit_per_validator(
                    metagraph=self.validator.metagraph,
                    epoch_volume=info.epoch_volume,
                    min_stake=self.validator.config.min_stake,
                    log=False,
                )
                info.rate_limit = rate_limit_per_validator.get(
                    self.validator.uid, MIN_RATE_LIMIT
                )
                info.reward_scale = max(min(info.epoch_volume / 512, 1), 0)
                self.all_uids_info[int(uid)] = info
                miner_distribution.setdefault(info.category, []).append(uid)

                bt.logging.info(f"Rate limit for {uid}: {info.rate_limit}")
            for category, uids in miner_distribution.items():
                bt.logging.info(f"{len(uids)} Miners in category {category}: {uids}")
            bt.logging.success("Updated miner identity")
            return True
        except Exception as e:
            bt.logging.error(f"Update miner identity error: {e}")
            traceback.print_exc()
            return False

    def get_miner_uids(self, category: str):
        """
        Get miner uids based on category, useful if subnet has multiple categories
        """
        print(self.all_uids_info)
        available_uids = [
            int(uid)
            for uid in self.all_uids_info.keys()
            if self.all_uids_info[uid].category == category
        ]
        return available_uids

    def update_scores(self, uids, rewards):
        """
        Update miner's scores with new rewards
        """
        for uid, reward in zip(uids, rewards):
            self.all_uids_info[uid].scores.append(reward)
            self.all_uids_info[uid].scores = self.all_uids_info[uid].scores[
                -NO_OF_RECENT_SCORES:
            ]

    def get_on_chain_weights(self, category) -> torch.Tensor:
        """
        Get on-chain weights for miners based on their scores, do some normalization and clipping. Useful when have multiple categories
        """
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

    def get_model_specific_weights(self, category, normalize=True):
        """
        Get model specific weights for miners running this model based on their scores, do some normalization and clipping. Useful when have multiple categories
        """
        model_specific_weights = torch.zeros(len(self.all_uids))
        for uid in self.get_miner_uids(category):
            num_past_to_check = 10
            model_specific_weights[int(uid)] = (
                sum(self.all_uids_info[uid].scores[-num_past_to_check:])
                / num_past_to_check
            )
        model_specific_weights = torch.clamp(model_specific_weights, 0, 1)
        if normalize:
            tensor_sum = torch.sum(model_specific_weights)
            # Normalizing the tensor
            if tensor_sum > 0:
                model_specific_weights = model_specific_weights / tensor_sum
        return model_specific_weights
