import time
import bittensor as bt


def check_min_stake(stake: float, validator_uid: int, min_stake: float):
    return stake < min_stake


def check_limit(
    self, uid: str, stake: int, volume_per_validator: dict, interval: int = 600
):
    bt.logging.info(self.validator_logs)

    # Get the current max_request for the validator
    max_request = volume_per_validator.get(uid, 1)

    if uid not in self.validator_logs:
        self.validator_logs[uid] = {
            "start_interval": time.time(),
            "max_request": max_request,
            "request_counter": 1,
        }
    else:
        # Update max_request in case it has changed
        self.validator_logs[uid]["max_request"] = max_request

        if time.time() - self.validator_logs[uid]["start_interval"] > interval:
            self.validator_logs[uid]["start_interval"] = time.time()
            self.validator_logs[uid]["request_counter"] = 1
            bt.logging.info(f"Resetting counting log for uid: {uid}")
        else:
            self.validator_logs[uid]["request_counter"] += 1

    # Log the current state for debugging
    bt.logging.info(f"{self.validator_logs}")

    if self.validator_logs[uid]["request_counter"] > self.validator_logs[uid]["max_request"]:
        return True  # Limit exceeded
    return False  # Within limit
