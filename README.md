<div align="center">

# Image Generating Subnet <!-- omit in toc -->

---

</div>

## Introduction
Welcome to the Image Generating Subnet project. This README provides an overview of the project's structure and example usage for both validators and miners.

### The Incentivized Internet
- [Discord](https://discord.gg/bittensor)
- [Network](https://taostats.io/)
- [Research](https://bittensor.com/whitepaper)

## Project Structure
- `image_generation_subnet`: Contains base, feature functions, and utilities for validators and miners.
- `neurons`: Contains the validator and miner loop.
- `dependency_modules`: Includes servers for `prompt_generation`, `rewarding`, and `market`.
- `scripts`: Contains scripts for running validators and miners.

## Example Usage
Before running the following commands, make sure to replace the placeholder arguments with appropriate values.

## Validator
- **Normal run**
```bash
python neurons/validator/validator.py \
--netuid 1 \
--subtensor.chain_endpoint ws://20.243.203.20:9946 \
--wallet.name validator --wallet.hotkey default \
--proxy.port 8080 \
--proxy.public_ip http://localhost \
--proxy.market_registering_url http://localhost:10003/get_credentials \  # the endpoint of dependency_modules/market/app.py
--reward_endpoint http://localhost:10002/verify \  # the endpoint of dependency_modules/rewarding/app.py
--prompt_generating_endpoint http://localhost:10001/prompt_generate  # the endpoint of dependency_modules/prompt_generating/app.py
```

- **pm2 run**
1. Install [`pm2`](https://pm2.keymetrics.io/docs/usage/quick-start/) and [`jq`](https://jqlang.github.io/jq/download/)
2. Execute
```bash
pm2 start run.sh --attach
```

## Miner
To start a miner, you need to first set up an image generation endpoint on a GPU server, and then you can attach multiple miners to that endpoint.

For faster generation, you can have the miners on the same server as you have the GPU that runs the image generation endpoint.


### Starting Image Generation Endpoint

#### Install Dependencies
`pip install -r neurons/miner/example_endpoint/requirements.txt`

#### Download Model
`sh neurons/miner/example_endpoint/download_checkpoint.sh`

#### Start Image Generation Endpoint
`python3 neurons/miner/example_endpoint/app.py --port 10006`


#### Start Miner
```bash 
python -m neurons.miner.miner \
--netuid 1 \
--subtensor.chain_endpoint ws://20.243.203.20:9946 \
--wallet.name miner --wallet.hotkey default \
--generate_endpoint http://127.0.0.1:10006/generate \ # the endpoint of neurons/miner/example_endpoint
--info_endpoint http://127.0.0.1:10006/info \ # the endpoint of neurons/miner/example_endpoint
--axon.port 45467 # public port on the host machine
```

### Starting Miner That Uses The Image Generation Endpoint
```bash
python neurons/miner/miner.py \
--netuid 1 \
--subtensor.chain_endpoint ws://20.243.203.20:9946 \
--wallet.name miner --wallet.hotkey default \
--generate_endpoint http://127.0.0.1:10006/generate \ # the endpoint of neurons/miner/example_endpoint
--info_endpoint http://127.0.0.1:10006/info \ # the endpoint of neurons/miner/example_endpoint
--axon.port 12689 # public port on the host machine
```
