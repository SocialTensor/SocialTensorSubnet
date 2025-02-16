# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# import torch
import asyncio
import copy
from queue import Full
import threading
from datetime import datetime, timedelta, timezone
from traceback import print_exception
from typing import List

import bittensor as bt
import numpy as np
import requests

from image_generation_subnet.base.neuron import BaseNeuron
from image_generation_subnet.utils.weight_calculation import WeightCalculationService


class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    def __init__(self, config=None):
        super().__init__(config=config)
        self.weight_service = WeightCalculationService()

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Dendrite lets us send messages to other nodes (axons) in the network.
        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.scores = np.zeros_like(self.metagraph.S, dtype=np.float32)

        # Init sync with the network. Updates the metagraph.
        self.resync_metagraph()

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("axon off, not serving ip to chain.")

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()

    def serve_axon(self):
        """Serve axon to enable external connections."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.axon(wallet=self.wallet, config=self.config)

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")
                pass

        except Exception as e:
            bt.logging.error(f"Failed to create Axon initialize with exception: {e}")
            pass

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that validator is registered on the network.
        self.sync()

        bt.logging.info(
            f"Running validator on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        if hasattr(self, "axon"):
            f"Axon: {self.axon}"

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        while True:
            try:
                if self.config.proxy.port:
                    try:
                        self.validator_proxy.get_credentials()
                        bt.logging.info(
                            "Validator proxy ping to proxy-client successfully"
                        )
                    except Exception:
                        bt.logging.warning("Warning, proxy can't ping to proxy-client.")

                bt.logging.info(f"step({self.step}) block({self.block})")

                # Run forward.
                try:
                    self.forward()
                except Exception as err:
                    print_exception(type(err), err, err.__traceback__)

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()
                self.save_state()

                self.step += 1

            # If someone intentionally stops the validator, it'll safely terminate operations.
            except KeyboardInterrupt:
                self.axon.stop()
                bt.logging.success("Validator killed by keyboard interrupt.")
                exit()

            # In case of unforeseen errors, the validator will log the error and continue operations.
            except Exception as err:
                bt.logging.error("Error during validation", str(err))
                bt.logging.debug(print_exception(type(err), err, err.__traceback__))

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def get_bonus_scores(self):
        """
        Returns bonus scores for newly registered UIDs based on their registration date.
        Newer registrations get higher bonus percentages, scaling from 10% for 0-day-old
        registrations down to 1% for 9-day-old registrations.
        
        Returns:
            np.ndarray: Array of bonus scores matching the shape of self.scores
        """
        bonus_scores = np.zeros_like(self.scores)
        self.miner_manager.update_registration_log_from_api()
        try:
            days_since_registration_list = self._calculate_registration_days()
            bonus_scores = self._apply_bonus_multipliers(days_since_registration_list)
            bt.logging.info(f"Days since registration list: {days_since_registration_list}")
            
        except Exception as e:
            bt.logging.error(f"Error getting bonus scores: {e}")
            
        return bonus_scores

    def _calculate_registration_days(self):
        """
        Calculate days since registration for each UID.
        
        Returns:
            np.ndarray: Array containing days since registration for each UID
        """
        days_since_registration_list = np.zeros_like(self.scores)
        for uid in [int(uid) for uid in self.metagraph.uids]:
            try:
                registration_timestamp = self.miner_manager.registration_log[uid]
                days_since_registration = (datetime.now(timezone.utc) - datetime.fromisoformat(registration_timestamp).replace(tzinfo=timezone.utc)).days
                days_since_registration_list[uid] = days_since_registration

            except Exception as e:
                bt.logging.error(f"Error calculating registration days for uid {uid}: {e}")
                if uid < len(days_since_registration_list):
                    days_since_registration_list[uid] = 1000  # Ensures no bonus for this uid
                else:
                    bt.logging.error(f"Days since registration list is not large enough for uid {uid}")
                
        return days_since_registration_list

    def _apply_bonus_multipliers(self, days_since_registration_list: np.ndarray) -> np.ndarray:
        """
        Apply bonus multipliers based on days since registration.
        
        Args:
            days_since_registration_list: Array of days since registration for each UID
            
        Returns:
            np.ndarray: Array of bonus scores
        """
        bonus_scores = np.zeros_like(self.scores)
        bonus_percent_dict = {
            day: (10 - day) / 100  # Generates 0.10 to 0.01 for days 0-9
            for day in range(10)
        }
        
        for uid, days in enumerate(days_since_registration_list):
            if 0 <= days < 10:
                bonus_scores[uid] = bonus_percent_dict[int(days)] * self.scores[uid]
                
        return bonus_scores

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """
        # Add bonus scores to new registered uids
        bonus_scores = self.get_bonus_scores()
        bt.logging.info(f"Bonus scores: {bonus_scores}")
        self.scores = self.scores + bonus_scores

        # Check if self.scores contains any NaN values and log a warning if it does.
        if np.isnan(self.scores).any():
            bt.logging.warning(
                "Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        miner_raw_weights = np.nan_to_num(self.scores, nan=0)
        miner_raw_weight_sum = np.sum(np.abs(miner_raw_weights), axis=0, keepdims=True)
        if not miner_raw_weight_sum == 0:
            miner_raw_weights = miner_raw_weights / miner_raw_weight_sum
        bt.logging.info(f"Miner raw weights: {miner_raw_weights}")

        # Calculate weights base on alpha stake
        alpha_raw_weights = np.nan_to_num(self.metagraph.alpha_stake, nan=0)
        alpha_raw_weight_sum = np.sum(np.abs(alpha_raw_weights), axis=0, keepdims=True)
        if not alpha_raw_weight_sum == 0:
            alpha_raw_weights = alpha_raw_weights / alpha_raw_weight_sum
        bt.logging.info(f"Alpha raw weights: {alpha_raw_weights}")  

        # Calculate raw weights using the service
        raw_weights = self.weight_service.calculate_transition_weights(
            miner_raw_weights,
            alpha_raw_weights
        )
        bt.logging.info(f"Raw weights: {raw_weights}")
        bt.logging.trace("Top 10 values:", np.sort(raw_weights))
        bt.logging.trace("Top 10 uids:", np.argsort(raw_weights))

        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        bt.logging.trace("Processed weights:", processed_weights)
        bt.logging.trace("Processed weight uids:", processed_weight_uids)

         # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, 
            weights=processed_weights
        )
        bt.logging.debug("uint_weights", uint_weights)
        bt.logging.debug("uint_uids", uint_uids)

        # Set the weights on chain via our subtensor connection.
        result, log = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            version_key=self.spec_version,
        )

        if result:
            bt.logging.success(f"[SET WEIGHTS]: {log}")
        else:
            bt.logging.error(f"[SET WEIGHTS]: {log}")

        bt.logging.info(f"Set weights: {processed_weights}")

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = np.zeros((self.metagraph.n))
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    # def update_scores(self, rewards: torch.FloatTensor, uids: List[int]):
    #     """Performs exponential moving average on the scores based on the rewards received from the miners."""

    #     # Check if rewards contains NaN values.
    #     if torch.isnan(rewards).any():
    #         bt.logging.warning(f"NaN values detected in rewards: {rewards}")
    #         # Replace any NaN values in rewards with 0.
    #         rewards = torch.nan_to_num(rewards, 0)

    #     # Compute forward pass rewards, assumes uids are mutually exclusive.
    #     # shape: [ metagraph.n ]
    #     scattered_rewards: torch.FloatTensor = self.scores.scatter(
    #         0, torch.tensor(uids).to(self.device), rewards
    #     ).to(self.device)
    #     bt.logging.debug(f"Scattered rewards: {rewards}")

    #     # Update scores with rewards produced by this step.
    #     # shape: [ metagraph.n ]
    #     alpha: float = self.config.neuron.moving_average_alpha
    #     self.scores: torch.FloatTensor = alpha * scattered_rewards + (
    #         1 - alpha
    #     ) * self.scores.to(self.device)
    #     bt.logging.info(f"Updated moving avg scores: {self.scores}")