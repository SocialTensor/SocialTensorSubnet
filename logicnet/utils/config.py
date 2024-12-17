import os
import argparse
import bittensor as bt
from loguru import logger

MIN_STAKE = 10000


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

    # Neuron type
    parser.add_argument("--neuron_type", type=str, help="Neuron type", default="validator")

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

    parser.add_argument(
        "--wandb.off",
        action="store_true",
        help="Turn off wandb.",
        default=False,
    )

    parser.add_argument(
        "--wandb.project_name",
        type=str,
        default="logicnet-mainnet",
        help="Wandb project to log to.",
    )

    parser.add_argument(
        "--wandb.entity",
        type=str,
        default="ait-ai",
        help="Wandb entity to log to.",
    )

    # Neuron type
    neuron_type = "validator" if "miner" not in cls.__name__.lower() else "miner"

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
            "--neuron.vpermit_tao_limit",
            type=int,
            help="The maximum number of TAO allowed to query a validator with a vpermit.",
            default=4096,
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
            "--proxy.port",
            type=int,
            help="The port to run the proxy on.",
            default=None,
        )

        parser.add_argument(
            "--proxy.proxy_client_url",
            type=str,
            help="The url initialize credentials for proxy.",
            default="https://logicnet.aitprotocol.ai/proxy_client",
        )

        parser.add_argument(
            "--storage.storage_url",
            type=str,
            help="The url initialize to store miner's information.",
            default="https://logicnet.aitprotocol.ai/proxy_client/store_miner_information",
        )

        parser.add_argument(
            "--proxy.checking_probability",
            type=float,
            help="Probability of checking if a miner is valid",
            default=0.1,
        )

        parser.add_argument(
            "--min_stake",
            type=int,
            help="The minimum stake for a validator to be considered",
            default=MIN_STAKE,
        )

        parser.add_argument(
            "--llm_client.base_urls",
            type=str,
            help="The base url for the LLM client",
            default="http://localhost:8000/v1,https://api.openai.com/v1,https://api.together.xyz/v1",
        )

        parser.add_argument(
            "--llm_client.models",
            type=str,
            help="The model for the LLM client",
            default="Qwen/Qwen2-7B-Instruct,gpt-4o-mini,meta-llama/Llama-2-7b-chat-hf",
        )

        parser.add_argument(
            "--llm_client.keys",
            type=str,
            help="The key for the LLM client",
            default="xyz",
        )

        parser.add_argument(
            "--dataset_weight",
            type=str,
            help="The weight of the dataset",
            default="40,10,10,10,10,10,10",
        )

    else:
        parser.add_argument(
            "--miner.category",
            type=str,
            help="The category of the miner",
            default="Logic",
        )

        parser.add_argument(
            "--miner.total_volume",
            type=int,
            help="The total volume of requests to be served per 10 minutes",
            default=40,
        )

        parser.add_argument(
            "--miner.min_stake",
            type=int,
            help="The minimum stake for a validator to be considered",
            default=MIN_STAKE,
        )

        parser.add_argument(
            "--miner.limit_interval",
            type=int,
            help="The interval to limit the number of requests",
            default=600,
        )

        parser.add_argument(
            "--miner.llm_client.base_url",
            type=str,
            help="The base url for the LLM client",
            default="http://localhost:8000/v1",
        )

        parser.add_argument(
            "--miner.llm_client.model",
            type=str,
            help="The model for the LLM client",
            default="Qwen/Qwen2-7B-Instruct",
        )

        parser.add_argument(
            "--miner.llm_client.key",
            type=str,
            help="The key for the LLM client",
            default="xyz",
        )


def config(cls):
    """
    Returns the configuration object specific to this miner or validator after adding relevant arguments.
    """
    parser = argparse.ArgumentParser()
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)
    cls.add_args(parser)
    return bt.config(parser)