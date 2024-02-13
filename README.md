<div align="center">

# NicheImage - Image Generating Subnet <!-- omit in toc -->

---

</div>

## Introduction

Welcome to the NicheImage - A decentralized network of image models powered by the Bittensor protocol.

This README provides an overview of the project's structure and example usage for both validators and miners.


### NicheImage Studio ✨ - https://nicheimage.streamlit.app
![image](https://github.com/NicheTensor/NicheImage/assets/92072154/a02e299b-308d-40dd-90a2-5cc4789b896d)

### Bittensor Resources
- [Discord](https://discord.gg/bittensor)
- [Network Information](https://taostats.io/)
- [Bittensor Homepage](https://bittensor.com/)

## Project Structure
- `image_generation_subnet`: Contains base, feature functions, and utilities for validators and miners.
- `neurons`: Contains the validator and miner loop.
- `dependency_modules`: Includes servers for `prompt_generation`, `rewarding`, and `miner_endpoint`.

## Installation
1. Clone the repository.
```bash
git clone https://github.com/NicheTensor/NicheImage.git
```
2. Install the dependencies.
```bash
cd NicheImage
pip install -r requirements.txt
```
3. Install the project.
```bash
pip install -e .
```

## Example Usage
Before running the following commands, make sure to replace the placeholder arguments with appropriate values.

## Start Miner
Before running the following commands, make sure to replace the placeholder arguments with appropriate values.

First you need to start an image generation API on a gpu server that your miners can use. A RTX 3090 GPU is enough for several miners.
```
python dependency_modules/miner_endpoint/app.py -h
usage: app.py [-h] [--port PORT] [--model_name {RealisticVision,SDXLTurbo,AnimeV3,RealitiesEdgeXL}]

options:
  -h, --help            show this help message and exit
  --port PORT
  --model_name {RealisticVision,SDXLTurbo,AnimeV3,RealitiesEdgeXL}
```

```bash
python dependency_modules/miner_endpoint/app.py --port <port> --model_name <model_name>
```

You can also run with pm2. 
- SDXLTurbo:
```bash
pm2 start python --name "image_generation_endpoint_SDXLTurbo" -- -m dependency_modules.miner_endpoint.app --port 10006 --model_name SDXLTurbo
```
- RealitiesEdgeXL:
```bash
pm2 start python --name "image_generation_endpoint_RealitiesEdgeXL" -- -m dependency_modules.miner_endpoint.app --port 10006 --model_name RealitiesEdgeXL
```
- RealisticVision
```bash
pm2 start python --name "image_generation_endpoint_RealisticVision" -- -m dependency_modules.miner_endpoint.app --port 10006 --model_name RealisticVision
```
- AnimeV3
```bash
pm2 start python --name "image_generation_endpoint_AnimeV3" -- -m dependency_modules.miner_endpoint.app --port 10006 --model_name AnimeV3
```


Then you can run several miners using the image generation API:
```bash
pm2 start python --name "miner" \
-- \
-m neurons.miner.miner \
--netuid <netuid> \
--wallet.name <wallet_name> --wallet.hotkey <wallet_hotkey> \
--subtensor.network <network> \
--generate_endpoint <your_miner_endpoint>/generate \
--info_endpoint <your_miner_endpoint>/info \
--axon.port <your_public_port> \
```

You can also start with pm2, here is an example:
```bash
pm2 start python --name "miner" -- -m neurons.miner.miner --netuid 23 --wallet.name <wallet_name> --wallet.hotkey <wallet_hotkey> --subtensor.network finney --generate_endpoint http://127.0.0.1:10006/generate --info_endpoint http://127.0.0.1:10006/info --axon.port 10010
```

**View logs** 
```bash
pm2 logs miner
```

## Start Validator

Requirements: A validator only needs a cpu server to validate by using our free to use APIs for checking image generation. This is the default setting and requires no configuration.

However, it is possible to run your own image checking APIs if you prefer. This does require a GPU with min 20 GB of ram. You can see how to do this [here.](./dependency_modules/README.md)

If validators opt in to share their request capacity they will get paid for each image they generate. Opt in is done by specifying --proxy.port
If passed, a proxy server will start that allows us to query through their validator, and the validator will get paid weekly for the images they provide.

### Start Validator with Default Settings

```bash
pm2 start python --name "validator_nicheimage" \
-- -m neurons.validator.validator \
--netuid <netuid> \
--wallet.name <wallet_name> --wallet.hotkey <wallet_hotkey> \
--subtensor.network <network> \
--axon.port <your_public_port> \
--proxy.port <other_public_port> # Optional, pass only if you want allow queries through your validator and get paid
```

**View logs** 
```bash
pm2 logs validator_nicheimage
```

### Schedule update and restart validator
Pull the latest code from github and restart the validator every hour.
**Notice**, the validator must be named validator_nicheimage for the auto-updates to restart the process, so do not change the name from validator_nicheimage.
```bash
pm2 start auto_update.sh --name "auto-update"
```

# Roadmap

We will release updates on Tuesdays, in order to make it predictable for when changes to the network will be introduced. Furhter we will do our best to share updates in advance.

Here is the current roadmap for the subnet:


**13 Feb:** Upgrade model - SDXL Turbo will be gradually replaced by another high quality SDXL model

**27 Feb:** Adding Img2Img and ControlNet, updates to NicheImage Studio

**March:** Adding MidJourney to the network- one of the most advanced image generation engine

**April:** Adding open categories - where different models can compete and the best ones will win and remain on the network

