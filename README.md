<div align="center">

# NicheImage - Image Generating Subnet <!-- omit in toc -->

---

</div>

## Table of Contents
1. [Introduction](#introduction)
2. [Setup for miner](#miner)
3. [Setup for validator](#validator)

<div id='introduction'/>
## Introduction

Welcome to the NicheImage - A decentralized network of image models powered by the Bittensor protocol.

This README provides an overview of the project's structure and example usage for both validators and miners.

## What's news
- **27/2/2024** Release DreamShaper with Txt2Img, Img2Img and Controlnet features.

### NicheImage Studio ✨ - https://nicheimage.streamlit.app
![image](https://github.com/NicheTensor/NicheImage/assets/92072154/a02e299b-308d-40dd-90a2-5cc4789b896d)

### Bittensor Resources
- [Discord](https://discord.gg/bittensor)
- [Network Information](https://taostats.io/)
- [Bittensor Homepage](https://bittensor.com/)

### Overview
![nicheimage-v2 drawio (1)](https://github.com/NicheTensor/NicheImage/assets/92072154/6fede8e0-cf08-4da1-927f-17c512225961)

Above diagram is overview of how NicheImage Subnet does.
Let us explain more!

**Forward Pass**
```
For model_type in model_types:
  Validator gets synthentic requests for model type
  Validator sends these requests to miners running model_type
  Validator sends miners' request and response to Rewarding API, which uses **reproducing and hash matching mechanism to rewarding**
  Validator multiplies miners' score with `incentive_distribution` of model_type
Validator normalize scores
Validator sets weights on the chain
```

**Validator Proxy**
Validator can buy miner usage through our Proxy Client by creating a proxy

<div id='miner'/>

## Setup for miner
**Before becoming a miner**
- This subnet requires miner GPU to run Image Generating Model, it can be RTX 3090, RTX 4090, A100, A40,... The more stronger GPU, the more Incentive you get
- You have a wallet/hotkey registered to subnet 23
- Your machine has public ports
- Installed `python 3.10 or above`, `GPU` drivers
- These tutorials use pm2 to manage processes, this is optional but you can [learn to install](https://www.npmjs.com/package/pm2)

**Let's setup a miner**
1. Clone and install requirements
```
git clone https://github.com/NicheTensor/NicheImage
cd NicheImage
pip install -e .
```
2. Select a model type
 - RealisticVision: min 12GB VRAM, Stable Diffusion Architecture
 - DreamShaper: min 12GB VRAM, Stable Diffusion Architecture
 - AnimeV3: min 24GB VRAM, SDXL Architecture
 - RealitiesEdgeXL: min 24GB VRAM, SDXL Architecture 
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

<div id='validator'/>

## Setup for Validator
**Requirements** 
A validator only needs a cpu server to validate by using our free to use APIs for checking image generation. This is the default setting and requires no configuration.

However, it is possible to run your own image checking APIs if you prefer. This does require a GPU with min 20 GB of ram. You can see how to do this by the end of this section

If validators opt in to share their request capacity they will get paid for each image they generate. Opt in is done by specifying --proxy.port
If passed, a proxy server will start that allows us to query through their validator, and the validator will get paid weekly for the images they provide.

**Installation**

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

**Optional** Validator gets challenge prompt and image to make synthentic request from our endpoint as default, but you can setup your own server
1. Start prompt generating endpoint
```
pm2 start python --name "challenge_prompt" -- -m services.challenge_generating.app --port 11001 --disable_secure
```
2. Start image generating endpoint
```
pm2 start python --name "challenge_image" -- -m services.challenge_generating.app --port 11002 --disable_secure
```
**Optional** Validators gets reward from our endpoint as default, which uses reproducing and hash matching mechanism, but you can setup your own server

**Eg.**
- `AnimeV3`
```
pm2 start python --name "reward_AnimeV3" -- -m services.rewarding.app --port 12001 --model_name AnimeV3 --disable_secure
```
- `RealititesEdgeXL`
```
pm2 start python --name "reward_REXL" -- -m services.rewarding.app --port 12002 --model_name RealitiesEdgeXL --disable_secure
```
- `RealisticVision`
```
pm2 start python --name "reward_RV" -- -m services.rewarding.app --port 12003 --model_name RealisticVision --disable_secure
```
- `DreamShaper`
```
pm2 start python --name "reward_DreamShaper" -- -m services.rewarding.app --port 12004 --model_name DreamShaper --disable_secure
```
**Run validator with your own endpoints**
_replace localhost with another ip if you don't host locally_
```
pm2 start python --name "validator_nicheimage" \
-- -m neurons.validator.validator \
--netuid <netuid> \
--wallet.name <wallet_name> --wallet.hotkey <wallet_hotkey> \
--subtensor.network <network> \
--axon.port <your_public_port> \
--proxy.port <other_public_port> \ # Optional, pass only if you want allow queries through your validator and get paid
--challenge.prompt http://localhost:11001 \ 
--challenge.image http://localhost:11002 \
--reward_url.AnimeV3 http://localhost:12001 \
--reward_url.RealitiesEdgeXL http://localhost:12002 \
--reward_url.RealisticVision http://localhost:12003 \
--reward_url.DreamShaper http://localhost:12004 \
```
