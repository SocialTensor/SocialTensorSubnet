from datetime import datetime, timezone
import json
import time
import bittensor as bt
from image_generation_subnet.protocol import ImageGenerating, Information
import numpy as np
from image_generation_subnet.utils.volume_setting import get_volume_per_validator
import requests
from threading import Thread
import image_generation_subnet as ig_subnet

import typing_extensions
if typing_extensions.TYPE_CHECKING:
    from neurons.validator.validator import Validator


class MinerManager:
    def __init__(self, validator: "Validator", metagraph: "bt.metagraph"):
        self.validator = validator
        self.metagraph = metagraph
        self.all_uids = [int(uid.item()) for uid in self.metagraph.uids]
        self.all_uids_info = {
            uid: {"scores": [], "model_name": "", "process_time": []}
            for uid in self.all_uids
        }
        self.days_since_registration_dict = {
            uid: 0
            for uid in [int(uid.item()) for uid in self.metagraph.uids]
        }
        """ { uid: days since registration , ... }"""
        self.layer_one_axons = {}
    
    def update_days_since_registration_dict_from_api(self):
        try:
            registration_log_url = "https://nicheimage-api.nichetensor.com/registration_log"
            registration_log = requests.get(registration_log_url, timeout=10).json()
            # convert keys to int
            registration_log = {int(k): v for k, v in registration_log.items()}
            days_since_registration_dict = {
                uid: (datetime.now(timezone.utc) - datetime.fromisoformat(registration_timestamp).replace(tzinfo=timezone.utc)).days
                for uid, registration_timestamp in registration_log.items()
            }
            bt.logging.info(f"Days since registration dict: {days_since_registration_dict}")
            return days_since_registration_dict
        except Exception as e:
            bt.logging.error(f"Failed to get registration log: {e}")
            return self.days_since_registration_dict

    def get_miner_info(self, only_layer_one=False):
        """
        1. Query model_name of available uids
        """
        if only_layer_one:
            uids = self.layer_one_axons.keys()
            query_axons = [self.layer_one_axons[uid] for uid in uids]
        else:
            uids = [int(uid) for uid in self.validator.metagraph.uids]
            query_axons = [self.validator.metagraph.axons[uid] for uid in uids]
        synapse = Information()
        bt.logging.info("Requesting miner info using synapse Information")
        responses = self.validator.dendrite.query(
            query_axons,
            synapse,
            deserialize=False,
            timeout=60,
        )
        responses = {
            uid: response.response_dict for uid, response in zip(uids, responses)
        }
        if only_layer_one:
            bt.logging.debug(f"Some layer one miners: {list(responses.items())[:5]}")
        responses = {k: v for k, v in responses.items()}
        return responses

    def update_layer_zero(self, responses: dict):
        for uid, info in responses.items():
            is_layer_zero = info.get("is_layer_zero", False)
            is_layer_one = info.get("is_layer_one", False)
            if is_layer_zero:
                bt.logging.info(f"Layer zero: {uid}")
                axon = self.validator.metagraph.axons[uid]
                axon.ip = info["layer_one"]["ip"]
                axon.port = info["layer_one"]["port"]
                self.layer_one_axons[uid] = axon
            if uid in self.layer_one_axons and not is_layer_zero and not is_layer_one:
                self.layer_one_axons.pop(uid)
        bt.logging.success("Updated layer zero")

    def update_miners_identity(self):
        """
        1. Query model_name of available uids
        2. Update the available list
        """
        valid_miners_info = self.get_miner_info()
        self.update_layer_zero(valid_miners_info)
        layer_one_valid_miners_info = self.get_miner_info(only_layer_one=True)
        valid_miners_info.update(layer_one_valid_miners_info)

        self.days_since_registration_dict = self.update_days_since_registration_dict_from_api()

        if not valid_miners_info:
            bt.logging.warning("No active miner available. Skipping setting weights.")
        for uid, info in valid_miners_info.items():
            miner_state = self.all_uids_info.setdefault(
                uid,
                {"scores": [], "model_name": "", "process_time": []},
            )
            miner_state["registration_time"] = self.days_since_registration_dict.get(uid, None)
            model_name = info.get("model_name", "Recycle")
            if model_name == "Recycle":
                miner_state["scores"] = [0.9 ** self.days_since_registration_dict.get(uid, 1000)] * 10
            raw_volume = info.get("total_volume", 40)  # Default to 40 if not specified
            min_allowed_volume = 40
            max_allowed_volume = 256
            miner_state["total_volume"] = min(max(raw_volume, min_allowed_volume), max_allowed_volume)
            miner_state["min_stake"] = info.get("min_stake", 10000)
            miner_state["reward_scale"] = max(
                min(miner_state["total_volume"] ** 0.5 / 256**0.5, 1), 0
            )
            miner_state["device_info"] = info.get("device_info", {})

            volume_per_validator = get_volume_per_validator(
                self.validator.metagraph,
                miner_state["total_volume"],
                1.03,
                10000,
                False,
            )
            miner_state["rate_limit"] = volume_per_validator.get(self.validator.uid, 2)
            bt.logging.info(f"Rate limit for {uid}: {miner_state['rate_limit']}")
            if miner_state["model_name"] == model_name:
                continue
            miner_state["model_name"] = model_name
            miner_state["scores"] = []
            miner_state["process_time"] = []

        bt.logging.success("Updated miner identity")
        model_distribution = {}
        for uid, info in self.all_uids_info.items():
            model_distribution[info["model_name"]] = (
                model_distribution.get(info["model_name"], 0) + 1
            )
        # Remove all key type is str, keep only int from all_uids_info
        self.all_uids_info = {
            int(k): v for k, v in self.all_uids_info.items() if isinstance(k, int)
        }
        bt.logging.info(f"Model distribution: {model_distribution}")
        thread = Thread(target=self.store_miner_info, daemon=True)
        thread.start()

    def get_miner_uids(self, model_name: str):
        available_uids = [
            int(uid)
            for uid in self.all_uids_info.keys()
            if self.all_uids_info[uid]["model_name"] == model_name
        ]
        return available_uids

    def update_scores(self, uids, rewards):
        for uid, reward in zip(uids, rewards):
            self.all_uids_info[uid]["scores"].append(reward)
            self.all_uids_info[uid]["scores"] = self.all_uids_info[uid]["scores"][-10:]

    def update_metadata(self, uids, process_times):
        for uid, ptime in zip(uids, process_times):
            if "process_time" not in self.all_uids_info[uid]:
                self.all_uids_info[uid]["process_time"] = []
            self.all_uids_info[uid]["process_time"].append(ptime)
            self.all_uids_info[uid]["process_time"] = self.all_uids_info[uid][
                "process_time"
            ][-500:]

    def get_model_specific_weights(self, model_name, normalize=True):
        """
        Get the model specific weights for the given model name.
        """
        if model_name == "Stake_based":
            # Get UIDs where dividends are 0
            validator_uids = np.where(self.metagraph.dividends > 0)[0]
            alpha_stake = self.metagraph.alpha_stake
            alpha_stake[validator_uids] = 0 # Set validator's alpha stake to 0, only keep miner's alpha stake
            model_specific_weights = alpha_stake
        else:
            model_specific_weights = np.zeros(len(self.all_uids))
            uids = self.get_miner_uids(model_name)
            for uid in uids:
                num_past_to_check = 10
                model_specific_weights[int(uid)] = (
                    sum(self.all_uids_info[uid]["scores"][-num_past_to_check:])
                    / num_past_to_check
                )
            model_specific_weights = np.clip(model_specific_weights, a_min=0, a_max=1)

        if model_name != "Recycle" and model_name != "Stake_based":
            bonus_scores = self.get_bonus_scores(uids, model_specific_weights)
            model_specific_weights = model_specific_weights + bonus_scores
            bt.logging.info(f"Bonus scores for {model_name}: {bonus_scores}")

        if normalize:
            array_sum = np.sum(model_specific_weights)
            # Normalizing the tensor
            if array_sum > 0:
                model_specific_weights = model_specific_weights / array_sum
        return model_specific_weights

    def store_miner_info(self):
        catalogue = {}
        for k, v in self.validator.nicheimage_catalogue.items():
            catalogue[k] = {
                "model_incentive_weight": v.get("model_incentive_weight", 0),
                "supporting_pipelines": v.get("supporting_pipelines", []),
            }
        data = {
            "uid": self.validator.uid,
            "info": self.all_uids_info,
            "version": ig_subnet.__version__,
            "catalogue": catalogue,
        }
        serialized_data = json.dumps(data, sort_keys=True, separators=(',', ':'))
        nonce = str(time.time_ns())
        # Calculate validator 's signature
        keypair = self.validator.wallet.hotkey
        message = f"{serialized_data}{keypair.ss58_address}{nonce}"
        signature = f"0x{keypair.sign(message).hex()}"
        # Add validator 's signature
        data["nonce"] = nonce
        data["signature"] = signature
        try:
            requests.post(
                self.validator.config.storage_url + "/store_miner_info",
                json=data
            )
            self.reset_metadata()
        except Exception as e:
            bt.logging.error(f"Failed to store miner info: {e}")

    def reset_metadata(self):
        for uid in self.all_uids_info:
            self.all_uids_info[uid]["process_time"] = []

    def get_bonus_scores(self, uids, model_specific_weights):
        """
        Returns bonus scores for newly registered UIDs based on their registration date.
        Newer registrations get higher bonus percentages, scaling from 10% for 0-day-old
        registrations down to 1% for 9-day-old registrations.
        
        Returns:
            np.ndarray: Array of bonus scores matching the shape of self.scores
        """
        bonus_scores = np.zeros_like(model_specific_weights)
        bonus_percent_dict = {
            day: (10 - day) / 100  # Generates 0.10 to 0.01 for days 0-9
            for day in range(10)
        }

        try:
            for uid in uids:
                days = self.days_since_registration_dict[uid]
                if days < 10:
                    bonus_scores[uid] = bonus_percent_dict[days] * model_specific_weights[uid]
            
        except Exception as e:
            bt.logging.error(f"Error getting bonus scores: {e}")
            
        return bonus_scores