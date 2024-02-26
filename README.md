## NicheImage - Decentralized Image Generation Network README

### Table of Contents
1. [Introduction](#introduction)
2. [Setup Instructions for Miners](#setup-for-miners)
3. [Setup Instructions for Validators](#setup-for-validators)

<div id='introduction'/>

### Introduction

NicheImage is a decentralized network that utilizes the Bittensor protocol to enable distributed image generation. This document serves as a guide for setting up and participating in the network, targeting both validators and miners. It includes essential information on project setup, operation, and contribution.

#### Latest Updates
- **27/2/2024**: Introduction of DreamShaper featuring Txt2Img, Img2Img, and Controlnet capabilities.

#### NicheImage Studio - [Visit Here](https://nicheimage.streamlit.app)
![image](https://github.com/NicheTensor/NicheImage/assets/92072154/a02e299b-308d-40dd-90a2-5cc4789b896d)

#### Additional Resources
- [Join Our Discord](https://discord.gg/bittensor)
- [Network Statistics](https://taostats.io/)
- [Learn More About Bittensor](https://bittensor.com/)

#### System Architecture Overview
![nicheimage-v2 drawio (1)](https://github.com/NicheTensor/NicheImage/assets/92072154/6fede8e0-cf08-4da1-927f-17c512225961)

### How the Subnet Operates
**Forward Pass Workflow:**
```
1. Validators receive synthetic requests for various model types.
2. These requests are forwarded to miners specializing in the requested models.
3. Responses from miners, along with request details, are sent to the Rewarding API.
   The API employs a "reproducing and hash matching" mechanism for miner rewards.
4. Miner scores are adjusted based on the incentive distribution for their model type.
5. Validators normalize these scores and update the blockchain with new weights.
```

**Validator Proxy:**
- Validators can utilize a Proxy Client to access miner services, allowing for a seamless integration within the network.

<div id='setup-for-miners'/>
  
### Setup Instructions for Miners
**Pre-requisites:**
- A powerful GPU (e.g., RTX 3090, RTX 4090, A100, A40) is required for image generation tasks.
- Ensure you have a registered wallet/hotkey for subnet 23, accessible public ports, and Python 3.10+ with GPU drivers installed.
- These tutorials use pm2 to manage processes, this is optional but you can [learn to install](https://www.npmjs.com/package/pm2)

**Setup step by step**
1. Clone and install requirements
```
git clone https://github.com/NicheTensor/NicheImage
cd NicheImage
pip install -e .
```
2. Select a model type
 - `RealisticVision`: min 12GB VRAM, Stable Diffusion Architecture
 - `DreamShaper`: min 12GB VRAM, Stable Diffusion Architecture
 - `AnimeV3`: min 24GB VRAM, SDXL Architecture
 - `RealitiesEdgeXL`: min 24GB VRAM, SDXL Architecture 
3. Self host a generation endpoint for `seleted_model_type` at step 2
```
pm2 start python --name "miner_endpoint" -- -m services.miner_endpoint.app --port 10006 --model_name <selected_model_type> --num_replicas <num_replicas> --num_gpus <num_gpus_per_replica>
```
This script uses [Ray](ray.io) allow you to run multi replicas based on your GPU hardware. This will be benefit because validators will penalize for time to response or be timed out will receive 0 reward.

**Eg.**
- If you have 1 GPU RTX 4090 and selected AnimeV3
```
pm2 start python --name "miner_endpoint" -- -m services.miner_endpoint.app --port 10006 --model_name Anime --num_replicas 1 --num_gpus 1
```
- If you have 2 GPUs RTX 4090 and selected AnimeV3, you can serve 2 replicas, each replica is on 1 GPU
```
pm2 start python --name "miner_endpoint" -- -m services.miner_endpoint.app --port 10006 --model_name Anime --num_replicas 2 --num_gpus 2
```
- If you have 1 GPU A100-80G and selected AnimeV3, you can serve 3 replicas, each allocated ~26.6GB VRAM to optimize your large VRAM
```
pm2 start python --name "miner_endpoint" -- -m services.miner_endpoint.app --port 10006 --model_name Anime --num_replicas 3 --num_gpus 0.33
```
4. Run a miner
```
pm2 start python --name "miner" \
-- \
-m neurons.miner.miner \
--netuid 23 \
--wallet.name <wallet_name> --wallet.hotkey <wallet_hotkey> \
--subtensor.network <network> \
--axon.port <your_public_port> \
--generate_endpoint http//:127.0.0.1:10006/generate \
--info_endpoint http//:127.0.0.1:10006/info \
```

<div id='setup-for-validators'/>

### Setup Instructions for Validators
**Requirements:**

A validator only needs a cpu server to validate by using our free to use APIs for checking image generation. This is the default setting and requires no configuration.

However, it is possible to run your own image checking APIs if you prefer. This does require a GPU with min 20 GB of ram. You can see how to do [here](./services/README.md)

If validators opt in to share their request capacity they will get paid for each image they generate. Opt in is done by specifying --proxy.port
If passed, a proxy server will start that allows us to query through their validator, and the validator will get paid weekly for the images they provide.

**Installation:**

1. Clone and install requirements
```
git clone https://github.com/NicheTensor/NicheImage
cd NicheImage
pip install -e .
```
2. Run validator
```
pm2 start python --name "validator_nicheimage" \
-- -m neurons.validator.validator \
--netuid <netuid> \
--wallet.name <wallet_name> --wallet.hotkey <wallet_hotkey> \
--subtensor.network <network> \
--axon.port <your_public_port> \
--proxy.port <other_public_port> # Optional, pass only if you want allow queries through your validator and get paid
```

