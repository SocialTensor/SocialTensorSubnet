import bittensor as bt
import torch

MIN_RATE_LIMIT = 2


def get_rate_limit_per_validator(
    metagraph,
    epoch_volume: int,
    min_stake: int,
    log: bool = True,
) -> dict:
    """
    Calculate the rate limit for each validator based on the epoch volume and the stake of the validators.
    The rate limit is the number of requests that a validator can process in a single epoch.
    """
    all_stakes = [stake for stake in metagraph.total_stake.tolist()]
    all_uids = [uid for uid in range(len(all_stakes))]
    valid_stakes = [stake for stake in all_stakes if stake >= min_stake]
    valid_uids = [uid for uid, stake in zip(all_uids, all_stakes) if stake >= min_stake]

    if not valid_stakes:
        bt.logging.warning(
            (
                f"No validators with stake greater than {min_stake} found. "
                "Assigning equal volume to all validators."
                f"Total volume: {epoch_volume}"
                f"Metagraph stake: {metagraph.total_stake.tolist()}"
            )
        )
        valid_uids = all_uids
        valid_stakes = [0] * len(all_stakes)
        min_stake = 0

    valid_stakes = torch.tensor(valid_stakes) + 1e-4
    normalized_valid_stakes = valid_stakes / valid_stakes.sum()
    volume_per_validator = epoch_volume * normalized_valid_stakes
    volume_per_validator = torch.floor(volume_per_validator)
    volume_per_validator = dict(zip(valid_uids, volume_per_validator.tolist()))
    for uid, volume in volume_per_validator.items():
        if metagraph.total_stake[uid] >= min_stake:
            volume_per_validator[uid] = max(MIN_RATE_LIMIT, volume)
        if log:
            bt.logging.info(
                f"Volume for {uid}-validator: stake: {metagraph.total_stake[uid]}, volume: {volume_per_validator[uid]}"
            )

    return volume_per_validator
