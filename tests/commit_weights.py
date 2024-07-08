from neurons.validator.validator import Validator
import argparse
import numpy as np
import bittensor.utils.weight_utils as weight_utils
import os

val = Validator()

n_uids = len(val.metagraph)
uids = np.arange(0, n_uids, dtype=int)
weights = np.ones(n_uids, dtype=float)

processed_uids, processed_weights = weight_utils.convert_weights_and_uids_for_emit(uids, weights)

salt_length = 8
salt = list(os.urandom(salt_length))

# Attempt to commit the weights to the blockchain.
success, msg = subtensor.commit_weights(wallet=val.wallet, netuid=val.config.netuid, salt=salt, uids=processed_uids, weights=processed_weights, wait_for_inclusion=True, wait_for_finalization=True) 


print(success, msg)
