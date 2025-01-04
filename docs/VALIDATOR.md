# LogicNet: Validator Documentation

## Overview

The Validator is responsible for generating challenges for the Miner to solve. It evaluates solutions submitted by Miners and rewards them based on the quality and correctness of their answers. Additionally, it incorporates penalties for late responses.

**Protocol**: `LogicSynapse`

- **Validator Prepares**:
  - `raw_logic_question`: A math problem generated using MathGenerator.
  - `logic_question`: A personalized challenge created by refining `raw_logic_question` with an LLM.
- **Miner Receives**:
  - `logic_question`: The challenge to solve.
- **Miner Submits**:
  - `logic_reasoning`: Step-by-step reasoning to solve the challenge.
  - `logic_answer`: The final answer to the challenge, expressed as a short sentence.

### Reward Structure

- **Correctness (`bool`)**: Checks if `logic_answer` matches the ground truth.
- **Similarity (`float`)**: Measures cosine similarity between `logic_reasoning` and the Validator’s reasoning.
- **Time Penalty (`float`)**: Applies a penalty for delayed responses based on the formula:
  
  ```
  time_penalty = (process_time / timeout) * MAX_PENALTY
  ```

## Setup for Validator

Follow the steps below to configure and run the Validator.

### Step 1: Configure for vLLM

This setup allows you to run the Validator locally by hosting a vLLM server. While it requires significant resources, it offers full control over the environment.

#### Minimum Compute Requirements

- **GPU**: 1x GPU with 24GB VRAM (e.g., RTX 4090, A100, A6000)
- **Storage**: 100GB
- **Python**: 3.10

#### Steps

1. **Set Up vLLM Environment**
   ```bash
   python -m venv vllm
   . vllm/bin/activate
   pip install vllm
   ```

2. **Install PM2 for Process Management**
   ```bash
   sudo apt update && sudo apt install jq npm -y
   sudo npm install pm2 -g
   pm2 update
   ```

3. **Select a Model**
   Supported models are listed [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

4. **Start the vLLM Server**
   ```bash
   . vllm/bin/activate
   pm2 start "vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --host 0.0.0.0" --name "sn35-vllm"
   ```
   *Adjust the model, port, and host as needed.*
   eg. include this if the model fail to start `--max-model-len 16384 --gpu-memory-utilization 0.95` 

---

### Step 2: Configure for Together AI and Open AI

Using Together AI and Open AI simplifies setup and reduces local resource requirements. At least one of these platforms must be configured.

#### Prerequisites

- **Account on Together.AI**: [Sign up here](https://together.ai/).
- **Account on Hugging Face**: [Sign up here](https://huggingface.co/).
- **API Key**: Obtain from the Together.AI dashboard.
- **Python 3.10**
- **PM2 Process Manager**: For running and managing the Validator process. *OPTIONAL*

#### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/LogicNet-Subnet/LogicNet logicnet
   cd logicnet
   ```

2. **Install the Requirements**
   ```bash
   python -m venv main
   . main/bin/activate

   bash install.sh
   ```
   Alternatively, install manually:
   ```bash
   pip install -e .
   pip uninstall uvloop -y
   pip install git+https://github.com/lukew3/mathgenerator.git
   ```

3. **Set Up the `.env` File**
   ```bash
   echo "TOGETHERAI_API_KEY=your_together_ai_api_key" > .env
   echo "OPENAI_API_KEY=your_openai_api_key" >> .env
   echo "HF_TOKEN=your_hugging_face_token" >> .env (needed for some vLLM model)
   echo "USE_TORCH=1" >> .env
   ```

### Step 3: Run the Validator

1. **Activate Virtual Environment**
   ```bash
   . main/bin/activate
   ```

2. **Source the `.env` File**
   ```bash
   source .env
   ```

3. **Start the Validator**
| You must run at least 2 models in any combination of 3
   ```bash
   pm2 start python --name "sn35-validator" -- neurons/validator/validator.py \
     --netuid 35 \
     --wallet.name "your-wallet-name" \
     --wallet.hotkey "your-hotkey-name" \
     --subtensor.network finney \
     --llm_client.base_urls "http://localhost:8000/v1,https://api.openai.com/v1,https://api.together.xyz/v1" \
     --llm_client.models "Qwen/Qwen2.5-7B-Instruct,gpt-4o-mini,meta-llama/Llama-3.3-70B-Instruct-Turbo" \
     --neuron_type validator \
     --logging.debug
   ```

4. **Enable Public Access (Optional)**
   Add this flag to enable proxy:
   ```bash
   --axon.port "your-public-open-port"
   ```

---

### Additional Features

#### Wandb Integration

Configure Wandb to track and analyze Validator performance.

1. Add Wandb API key to `.env`:
   ```bash
   echo "WANDB_API_KEY=your_wandb_api_key" >> .env
   ```
2. It's already configured for mainnet as default.
3. Run Validator with Wandb on Testnet:
   ```bash
   --wandb.project_name logicnet-testnet \
   --wandb.entity ait-ai
   ```

---

### Troubleshooting & Support

- **Logs**:
  - Please see the logs for more details using the following command.
  ```bash
  pm2 logs sn35-validator
  ```
  - Please check the logs for more details on wandb for mainnet.
    https://wandb.ai/ait-ai/logicnet-mainnet/runs
  - Please check the logs for more details on wandb for testnet.
    https://wandb.ai/ait-ai/logicnet-testnet/runs

- **Common Issues**:
  - Missing API keys.
  - Incorrect model IDs.
  - Connectivity problems.
- **Contact Support**: Reach out to the LogicNet team for assistance.

---

Happy Validating!
