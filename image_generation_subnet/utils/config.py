# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

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

import os
import argparse
import bittensor as bt
from loguru import logger


def check_config(cls, config: "bt.Config"):
    r"""Checks/validates the config namespace object."""
    bt.logging.check_config(config)

    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,  # TODO: change from ~/.bittensor/miners to ~/.bittensor/neurons
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )
    print("full path:", full_path)
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)

    if not config.neuron.dont_save_events:
        # Add custom event logger for the events.
        logger.level("EVENTS", no=38, icon="📝")
        logger.add(
            os.path.join(config.neuron.full_path, "events.log"),
            rotation=config.neuron.events_retention_size,
            serialize=True,
            enqueue=True,
            backtrace=False,
            diagnose=False,
            level="EVENTS",
            format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        )


def add_args(cls, parser):
    """
    Adds relevant arguments to the parser for operation.
    """
    # Netuid Arg: The netuid of the subnet to connect to.
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=1)

    neuron_type = "validator" if "miner" not in cls.__name__.lower() else "miner"

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
        default=neuron_type,
    )

    parser.add_argument(
        "--neuron.device",
        type=str,
        help="Device to run on.",
        default="cpu",
    )

    parser.add_argument(
        "--neuron.epoch_length",
        type=int,
        help="The default epoch length (how often we set weights, measured in 12 second blocks).",
        default=100,
    )

    parser.add_argument(
        "--neuron.events_retention_size",
        type=str,
        help="Events retention size.",
        default="2 GB",
    )

    parser.add_argument(
        "--neuron.dont_save_events",
        action="store_true",
        help="If set, we dont save events to a log file.",
        default=False,
    )

    if neuron_type == "validator":
        parser.add_argument(
            "--neuron.disable_set_weights",
            action="store_true",
            help="Disables setting weights.",
            default=False,
        )

        parser.add_argument(
            "--neuron.axon_off",
            "--axon_off",
            action="store_true",
            # Note: the validator needs to serve an Axon with their IP or they may
            #   be blacklisted by the firewall of serving peers on the network.
            help="Set this flag to not attempt to serve an Axon.",
            default=False,
        )

        parser.add_argument(
            "--loop_base_time",
            type=int,
            help="The base time for the loop to run in seconds.",
            default=600,
        )

        parser.add_argument(
            "--async_batch_size",
            type=int,
            help="The number of threads to run in a single loop.",
            default=16,
        )

        parser.add_argument(
            "--storage_url",
            type=str,
            help="The url to store the image.",
            default="https://nicheimage-api.nichetensor.com/storage",
        )

        parser.add_argument(
            "--proxy.port",
            type=int,
            help="The port to run the proxy on.",
            default=None,
        )

        parser.add_argument(
            "--proxy.proxy_client_url",
            type=str,
            help="The url initialize credentials for proxy.",
            default="https://proxy-client.nichetensor.com",
        )

        parser.add_argument(
            "--proxy.checking_probability",
            type=float,
            help="Probability of checking if a miner is valid",
            default=0.1,
        )

        parser.add_argument(
            "--reward_url.RealitiesEdgeXL",
            type=str,
            default="https://nicheimage-api.nichetensor.com/reward/RealitiesEdgeXL",
        )

        parser.add_argument(
            "--reward_url.AnimeV3",
            type=str,
            default="https://nicheimage-api.nichetensor.com/reward/AnimeV3",
        )

        parser.add_argument(
            "--reward_url.Gemma7b",
            type=str,
            help="The endpoint to get the reward for Gemma7b.",
            default="https://nicheimage-api.nichetensor.com/reward/Gemma7b",
        )

        parser.add_argument(
            "--reward_url.StickerMaker",
            type=str,
            help="The endpoint to get the reward for StickerMaker.",
            default="https://nicheimage-api.nichetensor.com/reward/StickerMaker",
        )

        parser.add_argument(
            "--reward_url.Llama3_70b",
            type=str,
            help="The endpoint to get the reward for Llama3_70b.",
            default="https://nicheimage-api.nichetensor.com/reward/Llama3_70b",
        )

        parser.add_argument(
            "--reward_url.Llama3_3_70b",
            type=str,
            help="The endpoint to get the reward for Llama3_3_70b.",
            default="https://nicheimage-api.nichetensor.com/reward/Llama3_3_70b",
        )

        parser.add_argument(
            "--reward_url.DeepSeek_R1_Distill_Llama_70B",
            type=str,
            help="The endpoint to get the reward for DeepSeek_R1_Distill_Llama_70B.",
            default="https://nicheimage-api.nichetensor.com/reward/DeepSeek_R1_Distill_Llama_70B",
        )

        parser.add_argument(
            "--reward_url.JuggernautXL",
            type=str,
            help="",
            default="https://nicheimage-api.nichetensor.com/reward/JuggernautXL",
        )

        parser.add_argument(
            "--reward_url.SUPIR",
            type=str,
            help="",
            default="https://nicheimage-api.nichetensor.com/reward/SUPIR",
        )
        parser.add_argument(
            "--reward_url.FluxSchnell",
            type=str,
            help="",
            default="https://nicheimage-api.nichetensor.com/reward/FluxSchnell",
        )

        parser.add_argument(
            "--reward_url.Kolors",
            type=str,
            help="",
            default="https://nicheimage-api.nichetensor.com/reward/Kolors",
        )

        parser.add_argument(
            "--reward_url.OpenCategory",
            type=str,
            help="",
            default="https://nicheimage-api.nichetensor.com/reward/OpenCategory",
        )

        parser.add_argument(
            "--reward_url.Pixtral_12b",
            type=str,
            help="",
            default="https://nicheimage-api.nichetensor.com/reward/Pixtral_12b",
        )

        # TODO: add more reward endpoints for categories

        parser.add_argument(
            "--challenge.prompt",
            type=str,
            help="The endpoint to send generate requests to.",
            default="https://nicheimage-api.nichetensor.com/challenge/prompt",
        )

        parser.add_argument(
            "--challenge.image",
            type=str,
            help="The endpoint to send generate requests to.",
            default="https://nicheimage-api.nichetensor.com/challenge/image",
        )

        parser.add_argument(
            "--challenge.llm_prompt",
            type=str,
            help="The endpoint to send generate requests to.",
            default="https://nicheimage-api.nichetensor.com/challenge/llm_prompt",
        )

        parser.add_argument(
            "--challenge.open_category_prompt",
            type=str,
            help="The endpoint to send generate requests to.",
            default="https://nicheimage-api.nichetensor.com/challenge/open_category_prompt",
        )

        parser.add_argument(
            "--challenge.visual_question_answering",
            type=str,
            help="The endpoint to send generate requests to.",
            default="https://nicheimage-api.nichetensor.com/challenge/vqa",
        )

        parser.add_argument(
            "--share_response",
            action="store_true",
            help="If set, validator will share miners' response to owner endpoint.",
            default=True,
        )

        parser.add_argument(
            "--offline_reward.enable",
            action="store_true",
            help="",
            default=False,
        )

        parser.add_argument(
            "--offline_reward.validator_endpoint",
            type=str,
            help="",
            default="http://127.0.0.1:13300/generate",
        )

        parser.add_argument(
            "--offline_reward.redis_endpoint",
            type=str,
            help="",
            default="http://127.0.0.1:6379",
        )

    else:
        parser.add_argument(
            "--blacklist.force_validator_permit",
            action="store_true",
            help="If set, we will force incoming requests to have a permit.",
            default=False,
        )

        parser.add_argument(
            "--blacklist.allow_non_registered",
            action="store_true",
            help="If set, miners will accept queries from non registered entities. (Dangerous!)",
            default=False,
        )

        parser.add_argument(
            "--generate_endpoint",
            type=str,
            help="The endpoint to send generate requests to.",
            default="http://127.0.0.1:10006/generate",
        )

        parser.add_argument(
            "--info_endpoint",
            type=str,
            help="The endpoint to send info requests to.",
            default="http://127.0.0.1:10006/info",
        )

        parser.add_argument(
            "--miner.total_volume",
            type=int,
            help="The total volume of requests to be served per 10 minutes",
            default=40,
        )

        parser.add_argument(
            "--miner.size_preference_factor",
            type=float,
            help="The size preference factor for the volume per validator",
            default=1.03,
        )

        parser.add_argument(
            "--miner.min_stake",
            type=int,
            help="The minimum stake for a validator to be considered",
            default=1000,
        )
        parser.add_argument(
            "--miner.limit_interval",
            type=int,
            help="The interval to limit the number of requests",
            default=600,
        )

        parser.add_argument(
            "--miner.max_concurrent_requests",
            type=int,
            help="The maximum number of concurrent requests to be served",
            default=4,
        )

        # LAYER ZERO

        parser.add_argument(
            "--miner.disable_serve_axon",
            action="store_true",
            help="If set, we will disable serving axon.",
            default=False,
        )

        parser.add_argument(
            "--miner.is_layer_zero",
            action="store_true",
            help="If set, we will enable layer zero.",
            default=False,
        )

        parser.add_argument(
            "--miner.is_layer_one",
            action="store_true",
            help="If set, we will enable layer one.",
            default=False,
        )

        parser.add_argument(
            "--miner.layer_one_ip",
            type=str,
            help="The endpoint to send generate requests to.",
            default="0.0.0.0",
        )

        parser.add_argument(
            "--miner.layer_one_port",
            type=int,
            help="The endpoint to send generate requests to.",
            default=8091,
        )


def config(cls):
    """
    Returns the configuration object specific to this miner or validator after adding relevant arguments.
    """
    parser = argparse.ArgumentParser()
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    # Set logging level to INFO
    bt.logging.before_enable_info()
    bt.logging.enable_info()
    bt.axon.add_args(parser)
    cls.add_args(parser)
    return bt.config(parser)
