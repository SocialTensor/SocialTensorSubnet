import os
import datetime
import wandb
import bittensor as bt
from dotenv import load_dotenv

from logicnet import __version__ as version

load_dotenv()

class WandbManager:
    def __init__(self, neuron=None):
        self.wandb = None
        self.wandb_start_date = datetime.date.today()
        self.neuron = neuron
        
        if not self.neuron.config.wandb.off:
            if os.getenv("WANDB_API_KEY"):
                self.init_wandb()
            else:
                bt.logging.warning("WANDB_API_KEY is not set. Please set it to use Wandb.") 
        else:
            bt.logging.warning("Running neuron without Wandb. Recommend to add Wandb.")

    def init_wandb(self):
        bt.logging.debug("Init wandb")
        
        """Creates a new wandb for neurons' logs"""
        self.wandb_start_date = datetime.date.today()
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        
        name = f"{self.neuron.neuron_type}-{self.neuron.uid}--{version}--{current_time}"
        # wandb init
        self.wandb = wandb.init(
            anonymous="allow",
            name=name,
            project="logicnet-testnet",
            entity="breo-workspace",
            config={
                "uid":self.neuron.uid,
                "hotkey":self.neuron.wallet.hotkey.ss58_address,
                "version":version,
                "type":self.neuron.config.neuron_type,
            }
        )
        
        bt.logging.info(f"Init a new Wandb: {name}")