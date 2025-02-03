import json
import time
import bittensor as bt
from image_generation_subnet.protocol import ImageGenerating, Information
import numpy as np
from image_generation_subnet.utils.volume_setting import get_volume_per_validator
import requests
from threading import Thread
import image_generation_subnet as ig_subnet


class MinerManager:
    def __init__(self, validator):
        self.validator = validator
        self.all_uids = [int(uid.item()) for uid in self.validator.metagraph.uids]
        self.all_uids_info = {
            uid: {"scores": [], "model_name": "", "process_time": []}
            for uid in self.all_uids
        }
        self.layer_one_axons = {}

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
        responses = {k: v for k, v in responses.items() if v}
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

        if not valid_miners_info:
            bt.logging.warning("No active miner available. Skipping setting weights.")
        for uid, info in valid_miners_info.items():
            miner_state = self.all_uids_info.setdefault(
                uid,
                {"scores": [], "model_name": "", "process_time": []},
            )
            model_name = info.get("model_name", "")
            miner_state["total_volume"] = info.get("total_volume", 40)
            miner_state["min_stake"] = info.get("min_stake", 10000)
            miner_state["reward_scale"] = max(
                min(miner_state["total_volume"] ** 0.5 / 1000**0.5, 1), 0
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
        model_specific_weights = np.zeros(len(self.all_uids))
        for uid in self.get_miner_uids(model_name):
            num_past_to_check = 10
            model_specific_weights[int(uid)] = (
                sum(self.all_uids_info[uid]["scores"][-num_past_to_check:])
                / num_past_to_check
            )
        model_specific_weights = np.clip(model_specific_weights, a_min=0, a_max=1)
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
