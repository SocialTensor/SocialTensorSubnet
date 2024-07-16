import time
import bittensor as bt


def check_min_stake(stake: float, validator_uid: int, min_stake: float):
    return stake < min_stake


def check_limit(
    self, uid: str, stake: int, volume_per_validator: dict, interval: int = 600
):
    bt.logging.info(self.validator_logs)

    if uid not in self.validator_logs:
        self.validator_logs[uid] = {
            "start_interval": time.time(),
            "max_request": volume_per_validator.get(uid, 1),
            "request_counter": 1,
        }
    elif time.time() - self.validator_logs[uid]["start_interval"] > interval:
        self.validator_logs[uid] = {
            "start_interval": time.time(),
            "max_request": volume_per_validator[uid],
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
