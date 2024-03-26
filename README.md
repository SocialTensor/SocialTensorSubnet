## NicheImage - Decentralized Image Generation Network README

### Table of Contents
1. [Introduction](#introduction)
2. [Setup Instructions for Miners](#setup-for-miners)
3. [Setup Instructions for Validators](#setup-for-validators)

<div id='introduction'/>

### Introduction

NicheImage is a decentralized network that utilizes the Bittensor protocol to enable distributed image generation. This document serves as a guide for setting up and participating in the network, targeting both validators and miners. It includes essential information on project setup, operation, and contribution.

**For more information, please click [here](image_generation_subnet/NicheImage.md)**
#### Latest Updates
- **5/3/2024**: 
   - Highlights
      - Introduction of DreamShaper featuring Txt2Img, Img2Img, and Controlnet capabilities.
      - Update incentive distribution
         - RealisticVision: 30%
         - DreamShaper: 6%
         - AnimeV3: 34%
         - RealitiesEdgeXL: 30%
      - Integrate Ray to allow multi replicas for each miner to optimize GPU usage.
      - Refractor the codebase
      - Better Safety Checker
   - Actions
      - Miners are required to update to use lastest codebase and Ray to optimize GPU usage.
         - It should be done by doing update and restart pm2 processes, or follow detailed instruction bellow.
      - Validators are required to update to have DreamShaper and new incentive distribution as well as new reward endpoint.
         - It should be done by doing update and restart pm2 processes, or follow detailed instruction bellow.

#### To Do
- [x] Add SOTA models for image generation
- [x] Integrate Ray to allow multi replicas for each miner to optimize GPU usage.
- [x] Add Img2Img and Controlnet capabilities to DreamShaper
- [ ] Add Img2Img and Controlnet capabilities to other models
- [ ] Add **Mid Journey**

#### NicheImage Studio - [Visit Here](https://nicheimage.streamlit.app)
![image](https://github.com/NicheTensor/NicheImage/assets/92072154/a02e299b-308d-40dd-90a2-5cc4789b896d)

#### Additional Resources
- [WandB Synthentic Dataset For Image Generation and Text Generation](https://wandb.ai/toilaluan/nicheimage)
- [Join Our Discord](https://discord.gg/bittensor)
- [Network Statistics](https://taostats.io/)
- [Learn More About Bittensor](https://bittensor.com/)

#### System Architecture Overview
![nicheimage-v2 drawio (1)](https://github.com/NicheTensor/NicheImage/assets/92072154/6fede8e0-cf08-4da1-927f-17c512225961)

### How the Subnet Operates
**Forward Pass Workflow:**
```
1. Validators create synthetic requests for various model types.
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
**Pre-requisites for GoJourney Model:**
- Only CPU needed
- Setup API KEY for [GoJourney](https://www.goapi.ai/midjourney-api)
- Ensure you have a registered wallet/hotkey for subnet 23, accessible public ports

**Pre-requisites for Stable Diffusion Model:**
- A powerful GPU (e.g., RTX 3090, RTX 4090, A100, A40) is required for image generation tasks.
- Ensure you have a registered wallet/hotkey for subnet 23, accessible public ports, and Python 3.10+ with GPU drivers installed.
- These tutorials use pm2 to manage processes, this is optional but you can [learn to install](https://www.npmjs.com/package/pm2)

**Setup step by step**
#### Clone and install requirements
```
git clone https://github.com/NicheTensor/NicheImage
cd NicheImage
pip install -e .
```
#### Select a model type
 - `RealisticVision`: min 12GB VRAM, Stable Diffusion Architecture
 - `DreamShaper`: min 12GB VRAM, Stable Diffusion Architecture
 - `AnimeV3`: min 24GB VRAM, SDXL Architecture
 - `RealitiesEdgeXL`: min 24GB VRAM, SDXL Architecture 
 - `GoJourney`: CPU only, MidJourney API based
#### Self host a generation endpoint for `seleted_model_type` at step 2

**Stable Diffusion Model**
```
pm2 start python --name "miner_endpoint" -- -m services.miner_endpoint.app --port 10006 --model_name <selected_model_type> --num_replicas <num_replicas> --num_gpus <num_gpus_per_replica>
```
This script uses [Ray](ray.io) allow you to run multi replicas based on your GPU hardware. This will be benefit because validators will penalize for time to response or be timed out will receive 0 reward.

**GoJourney Model**
```
GOJOURNEY_API_KEY=xxx PROCESS_MODE=yyy pm2 start python --name "miner_endpoint" -- -m services.miner_endpoint.app --port 10006 --model_name GoJourney --num_gpus 0
```
   - Get `GOJOURNEY_API_KEY` from [GoJourney](https://www.goapi.ai/midjourney-api)
   - `PROCESS_MODE` can be `relax`, `fast`, `turbo`. `relax` is slowest but only get 0.1 score each request, `turbo` is fastest, get 1.0 score each request. `fast` is in between, get 0.5 score each request

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


**Text Model: Gemma7B**

We're introducing a new text model category: `google/gemma-7b-it`. Mining this model is best done with an A100 series or RTX 4090 GPU.

To start mining with this model, follow these steps:

   1. **Install and Run vLLM:**
      - Create a new Python environment for `vLLM`:
      ```
      python -m venv vllm
      source vllm/bin/activate
      pip install vllm
      ```
      - Start the API server with your Hugging Face token (ensure access to `gemma-7b-it` at [https://huggingface.co/google/gemma-7b-it](https://huggingface.co/google/gemma-7b-it)):
      ```
      HF_TOKEN=YourHuggingFaceToken python -m vllm.entrypoints.openai.api_server --model google/gemma-7b-it
      ```

   2. **Install the NicheImage Repository:**
      ```
      git clone https://github.com/NicheTensor/NicheImage
      cd NicheImage
      python -m venv nicheimage
      source nicheimage/bin/activate
      pip install -e .
      ```

   3. **Run the Gemma7b Endpoint:**
      ```
      HF_TOKEN=YourHuggingFaceToken python services/miner_endpoint/text_app.py --model_name Gemma7b --port 10006
      ```



#### Run a miner

#### Setting Manual Generation Volume
- **Custom Generation Volume:** Miners now have the ability to specify their generation volume for every 10-minute interval. This volume will be distributed to validators based on the amount of TAO staked.
- **Configuration:** To set your generation volume, use the argument `--miner.total_volume X`, where `X` is your desired volume per 10 minutes.
- **Performance Metrics:** For instance, an RTX 4090 can process 1 RealisticVision Request per second, equating to 600 requests in 10 minutes.
- **Impact on Rewards:** The miner's maximum volume directly influences their rewards. The new reward formula is `new_reward = old_reward * (0.9 + 0.1 * volume_scale)`, where `volume_scale` is calculated as `max(min(total_volume ** 0.5 / 10, 1), 0)`.
- **Recommended Volume**: We recommend setting the volume to 150 to 200 for an RTX 4090 and A100

| `total_volume` | `volume_scale` |
|--------------|----------------|
| 100          | 1              |
| 50           | 0.707          |
| 30           | 0.547          |

- **Minimum Quota for Validators:** Any validator with a stake greater than 10,000 TAO is guaranteed a minimum quota of 2 requests/miner.

Example: If a miner sets `total_volume = 100` and there are two validators with stakes of 15K, 600K, 1M TAO respectively, the distribution of volume would be as follows:
- Validator 1 receives 2 requests because it has over 10k staked TAO.
- Validator 2 receives ~37 requests.
- Validator 3 receives ~62 requests

```
pm2 start python --name "miner" \
-- \
-m neurons.miner.miner \
--netuid 23 \
--wallet.name <wallet_name> --wallet.hotkey <wallet_hotkey> \
--subtensor.network <network> \
--axon.port <your_public_port> \
--generate_endpoint http://127.0.0.1:10006/generate \
--info_endpoint http://127.0.0.1:10006/info \
--miner.total_volume <your-generation-volume>
```

<div id='setup-for-validators'/>

### Setup Instructions for Validators
**Requirements:**

A validator only needs a cpu server to validate by using our free to use APIs for checking image generation. This is the default setting and requires no configuration.

However, it is possible to run your own image checking APIs if you prefer. This does require a GPU with min 20 GB of ram. You can see how to do [here](./services/README.md)

If validators opt in to share their request capacity they will get paid for each image they generate. Opt in is done by specifying --proxy.port
If passed, a proxy server will start that allows us to query through their validator, and the validator will get paid weekly for the images they provide.

**Integration with Weights & Biases:** Validators can now log miner responses to Weights & Biases. To enable this feature, add the `WANDB_API_KEY` from your Weights & Biases account settings ([https://wandb.ai/settings#api](https://wandb.ai/settings#api)) and set the `use_wandb` argument to true. Example command to start the validator with Weights & Biases logging enabled:


**Installation:**

1. Clone and install requirements
```
git clone https://github.com/NicheTensor/NicheImage
cd NicheImage
pip install -e .
```
2. Run validator
```
WANDB_API_KEY=YourWandBApiKey \ # optional
pm2 start python --name "validator_nicheimage" \
-- -m neurons.validator.validator \
--netuid <netuid> \
--wallet.name <wallet_name> --wallet.hotkey <wallet_hotkey> \
--subtensor.network <network> \
--axon.port <your_public_port> \
--proxy.port <other_public_port> # Optional, pass only if you want allow queries through your validator and get paid
--use_wandb # optional, log images and texts result as a dataset to WandB
```

