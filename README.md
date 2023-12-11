# Stable Diffusion Subnet

## Prerequisites Setup
Before proceeding, ensure that Bittensor is installed. Follow the instructions below:

- [Install `bittensor`](https://github.com/opentensor/bittensor#install).

Once `bittensor` is installed, proceed with the following steps:

### 1. Install Substrate Dependencies

Start by installing the necessary dependencies to run a Substrate node.

Update your system packages:

```bash
sudo apt update 
```

Install the required libraries and tools:

```bash
sudo apt install --assume-yes make build-essential git clang curl libssl-dev llvm libudev-dev protobuf-compiler
```

### 2. Install Rust and Cargo

Rust is the programming language used for Substrate development, and Cargo is the Rust package manager.

To install Rust and Cargo:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Update your shell's source to include Cargo's path:

```bash
source "$HOME/.cargo/env"
```

### 3. Clone the Subtensor Repository

Clone the Subtensor codebase to your local machine:

```bash
git clone https://github.com/toilaluan/subtensor.git
```

### 4. Setup & Run Subtensor Server

Navigate to the Subtensor directory and initialize the server:

```bash
cd subtensor
./scripts/init.sh
./scripts/localnet.sh
```

**NOTE**: Monitor the build and initialization outputs. The initial build may take some time, depending on your hardware.

### 5. Set Up Wallets

Different roles within the subnet, such as subnet owner, subnet validator, and subnet miner, require separate wallets:

- The owner wallet creates and manages the subnet.
- The validator and miner must register with the subnet created by the owner to run their respective scripts.

Create a coldkey for the owner role:

```bash
btcli wallet new_coldkey --wallet.name owner
```

Set up the miner's wallets:

```bash
btcli wallet new_coldkey --wallet.name miner
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default
```

Set up the validator's wallets:

```bash
btcli wallet new_coldkey --wallet.name validator
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default
```

### 6. Mint Tokens from Faucet

Tokens are required to initialize the incentive mechanism on the chain and for registering the subnet. 

- The owner needs > τ1000 to create a subnet.
- Validators & Miners need > τ0 to register with the subnet.

Mint faucet tokens for the owner, validator, and miner as follows:

```bash
# For the owner
btcli wallet faucet --wallet.name owner --subtensor.chain_endpoint ws://127.0.0.1:9946 
# Repeat as needed

# For the validator
btcli wallet faucet --wallet.name validator --subtensor.chain_endpoint ws://127.0.0.1:9946 

# For the miner
btcli wallet faucet --wallet.name miner --subtensor.chain_endpoint ws://127.0.0.1:9946 
```

### 7. Create a Subnet

Establish a new subnet on the local chain using the following command. The cost is exactly τ1000.000000000 for the first subnet:

```bash
btcli subnet create --wallet.name owner --subtensor.chain_endpoint ws://127.0.0.1:9946 
```

**NOTE**: The local chain assigns a default `netuid` of 1 for the first subnet. Subsequent registrations will increase the `netuid` sequentially. There is a limit of 8 subnets.

### 8. Register Keys

Register your subnet validator and miner on the subnet to allocate unique slots. The subnet currently supports up to 128 slots.

Commands for registering the subnet miner and validator:

```bash
# For the subnet miner
btcli subnet register --wallet.name miner --wallet.hotkey default --subtensor.chain_endpoint ws://127.0.0.1:9946

# For the subnet validator
btcli subnet register --wallet.name validator --wallet.hotkey default --subtensor.chain_endpoint ws://127.0.0.1:9946
```

### 11. Add Stake 

Bootstrap the incentives on your new subnet by adding stake to its incentive mechanism:

```bash
btcli stake add --wallet.name validator --wallet.hotkey default --subtensor.chain_endpoint ws://127.0.0.1:9946
```

## Setting Up the Stable Diffusion Subnet

- Validators iteratively request prompts from an LLM Server, then send requests to Miners for image generation.
- Miners generate images and return them.
- Validators request rewards from the

 Rewarding Server and update the weights.

### 0. Get the Stable Diffusion Subnet

Clone the necessary repositories and install dependencies:

```bash
git clone https://github.com/toilaluan/bittensor-fixed-imagenet.git
git submodule update --init --recursive
pip install -r requirements.txt
```

Install and run `Redis`:

```bash
sudo apt install redis
sudo systemctl restart redis
sudo systemctl status redis
```

### 1. Start Prompt & Reward API

Initialize the Prompt and Reward APIs:

```bash
# For Prompt API
cd prompt_gen_api
pip install -r requirements.txt
python app.py

# For Reward API
cd reward_api
pip install -r requirements.txt
python app.py
```

### 2. Run Subnet Miner and Validator

Define enviroment parameters `.env`, remember to change url & port based on your system:
```
PROMPT_ENDPOINT=http://127.0.0.1:15409/prompt_generate
REWARD_ENDPOINT=http://127.0.0.1:15410/verify
MINER_SD_ENDPOINT=http://127.0.0.1:15414/generate
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_LIST=bittensor
```

Execute the subnet miner and validator, ensuring the correct subnet parameters are specified:

```bash
# For the subnet miner
python sd_net/base_miner/miner.py --netuid 1 --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name miner --wallet.hotkey default --logging.debug

# For the subnet validator
python sd_net/validators/validator.py --netuid 1 --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name validator --wallet.hotkey default --logging.debug
```


## Run WebUI: Validator endpoint

We use streamlit for allowing user queue request (prompt) to `redis`, validator dequeues request from `redis` then do inference.

```bash
streamlit run sd_net/validators/webui.py
```