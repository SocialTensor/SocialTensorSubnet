import time
import bittensor as bt

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neurons.miner.miner import Miner

def check_min_stake(stake: float, validator_uid: int, min_stake: float):
    return stake < min_stake


def check_limit(
    self: "Miner",
    uid: str, stake: int, volume_per_validator: dict, interval: int = 600
) -> bool:
    """
    Function to check if the validator has exceeded the request limit (the volume of validators / max task per validator in the interval).

    If the validator is not in the validator_logs, initialize the validator_logs for the validator.
    
    Else, if the validator is in the validator_logs, check if the time interval has passed. If it has, reset the validator_logs for the validator.

    Else, increment the request counter, and check if the request counter has exceeded the max request. If it has, return True.

    Otherwise, return False.
    """
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
