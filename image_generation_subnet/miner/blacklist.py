import time
import bittensor as bt

CONFIG = {
    "blacklist": {
        "min_stake": 10000,
        "block_non_validator": True,
        "tao_based_limitation": {
            "tao_base_level": 10000,  # TAOs
            "interval": 600,  # seconds
            "max_requests_per_interval": 10,
        },
    }
}


def check_min_stake(stake: int, validator_uid: int):
    return stake < CONFIG["blacklist"]["min_stake"]


def calculate_max_request_per_interval(stake: int):
    return CONFIG["blacklist"]["tao_based_limitation"]["max_requests_per_interval"] * (
        stake // CONFIG["blacklist"]["tao_based_limitation"]["tao_base_level"]
    )


def check_limit(self, uid: str, stake: int):
    bt.logging.info(self.validator_logs)

    if uid not in self.validator_logs:
        self.validator_logs[uid] = {
            "start_interval": time.time(),
            "max_request": calculate_max_request_per_interval(stake=stake),
            "request_counter": 1,
        }
    elif (
        time.time() - self.validator_logs[uid]["start_interval"]
        > CONFIG["blacklist"]["tao_based_limitation"]["interval"]
    ):
        self.validator_logs[uid] = {
            "start_interval": time.time(),
            "max_request": calculate_max_request_per_interval(stake=stake),
            "request_counter": 1,
        }
        bt.logging.info(f"Reseting counting log for uid: {uid}")
    else:
        self.validator_logs[uid]["request_counter"] += 1
        if (
            self.validator_logs[uid]["request_counter"]
            > self.validator_logs[uid]["max_request"]
        ):
            return True
    return False
