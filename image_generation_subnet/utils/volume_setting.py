import bittensor as bt
import torch


def get_volume_per_validator(
    metagraph,
    total_volume: int,
    size_preference_factor: float,
    min_stake: int,
    log: bool = True,
) -> dict:
    all_stakes = [stake for stake in metagraph.total_stake.tolist()]
    all_uids = [uid for uid in range(len(all_stakes))]
    valid_stakes = [stake for stake in all_stakes if stake >= min_stake]
    valid_uids = [uid for uid, stake in zip(all_uids, all_stakes) if stake >= min_stake]
    if not valid_stakes:
        bt.logging.warning(
            (
                f"No validators with stake greater than {min_stake} found. "
                "Assigning equal volume to all validators."
                f"Total volume: {total_volume}"
                f"Metagraph stake: {metagraph.total_stake.tolist()}"
            )
        )
        valid_uids = all_uids
        valid_stakes = [0] * len(all_stakes)
        min_stake = 0

    valid_stakes = torch.tensor(valid_stakes) + 1e-4
    prefered_valid_stakes = valid_stakes**size_preference_factor
    normalized_prefered_valid_stakes = (
        prefered_valid_stakes / prefered_valid_stakes.sum()
    )
    volume_per_validator = total_volume * normalized_prefered_valid_stakes
    volume_per_validator = torch.floor(volume_per_validator)
    volume_per_validator = dict(zip(valid_uids, volume_per_validator.tolist()))
    for uid, volume in volume_per_validator.items():
        if metagraph.total_stake[uid] >= min_stake:
            volume_per_validator[uid] = max(2, volume)
        if log:
            bt.logging.info(
                f"Volume for {uid}-validator: stake: {metagraph.total_stake[uid]}, volume: {volume_per_validator[uid]}"
            )

    return volume_per_validator
