# NicheImage - Decentralized Image Generation Network

## Introduction

### Description
NicheImage is a decentralized network that utilizes the Bittensor protocol to enable distributed image generation. This document serves as a guide for setting up and participating in the network, targeting both validators and miners. It includes essential information on project setup, operation, and contribution.

For more information, please click [here](image_generation_subnet/NicheImage.md)

#### Try out the [NicheImage Generation Studio](https://nicheimage.streamlit.app/) for free image generation playground


### Incentive Distribution

| Category        | Incentive Distribution | Description                                                                                                        |
|-----------------|------------------------|--------------------------------------------------------------------------------------------------------------------|
| GoJourney       | 4%                     | API based - MidJourney Image Generation                                                                                        |
| AnimeV3         | 31%                    | SDXL Architecture                                                                                  |
| JuggernautXL | 17%                    | SDXL Architecture                                                            |
| RealitiesEdgeXL | 29%                    | SDXL Turbo Architecture                                                      |
| DreamShaperXL     | 6%                     | SDXL Architecture                           |
| Gemma7b         | 3%                     | Transformer LLM                                                     |
| Llama3_70b      | 4%                     | Transformer LLM|
| FaceToMany      | 3%                     | ComfyUI workflow, [https://replicate.com/fofr/face-to-many](https://replicate.com/fofr/face-to-many) |
| StickerMaker    | 3%                     | ComfyUI workflow, [https://replicate.com/fofr/sticker-maker](https://replicate.com/fofr/sticker-maker) |

### Key Features
- Designed to serve as a decentralized image generation network, incentivizing miners to scale up their computational resources. This allows a miner to contribute up to thousands of generations per minute to the network if they have sufficient GPU resources.
- Miners commit to a model type and generation volume.
- Miner's volume is distributed to validators based on the amount of TAO staked, with a minimum quota for validators.
- **Incentivized volume rewarding mechanism**: The reward formula incorporates the volume of requests handled, calculated as follows:
  - `new_reward = (matching_result - time_penalty) * (0.6 + 0.4 * volume_scale)`
  - `matching_result` is 0 or 1 based on the similarity matching result with the reproduction of the validator.
  - `time_penalty = 0.4 * (processing_time / timeout)**3`
  - `volume_scale = max(min(total_volume**0.5 / 1000**0.5, 1), 0)`
- Continual improvement of the network through the introduction of new models and features, such as the Gemma7b text model and the StickerMaker and FaceToMany models based on usage demand.
- Validators can earn money by sharing their request capacity with miners, allowing for seamless integration within the network.

## Setup and Participation

**These guide use `pm2` as a process manager**. If you don't have it installed, you can install it by following the instructions below:
- Install NodeJS Package Manager (npm) [here](https://nodejs.org/en/download/package-manager)
- Install pm2: `npm i pm2 -g`

### For Validators
1. Install the required dependencies and set up the NicheImage validator node.
2. Configure the validator settings, including the amount of TAO to stake.
3. Start the validator node and begin processing image generation requests from miners.

Detailed instructions on setting up the NicheImage validator node can be found [here](instructions/validator.md).

### For Miners
1. Install the necessary dependencies and set up the NicheImage miner node.
2. Choose the desired model type and specify the generation volume.
3. Start the miner node and begin contributing computational resources to the network.

Detailed instructions on setting up the NicheImage miner node can be found [here](instructions/miner.md).

## Contribution
We welcome contributions to the NicheImage project! If you have any ideas, bug reports, or feature requests, please open an issue on our GitHub repository. If you'd like to contribute code, please fork the repository and submit a pull request with your changes.
