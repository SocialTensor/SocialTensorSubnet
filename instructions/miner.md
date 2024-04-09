# Setup for miner

Make sure that you have a registered hotkey to Subnet 23. If you haven't done so, please refer to https://docs.bittensor.com/subnets/register-validate-mine


### Incentive Distribution

| Category        | Incentive Distribution | Timeout (s)                                                                                                        |
|-----------------|------------------------|--------------------------------------------------------------------------------------------------------------------|
| GoJourney       | 4%                     | 12 |
| AnimeV3         | 34%                    | 12 |
| RealisticVision | 20%                    | 12 |
| RealitiesEdgeXL | 30%                    | 12 |
| DreamShaper     | 6%                     | 12 |
| Gemma7b         | 3%                     | 64 |
| StickerMaker    | 3%                     | 64 |
<!-- | FaceToMany      | 1%                     | 48 | -->

## Step by Step Guide
1. Git clone and install requirements
```bash
git clone https://github.com/NicheTensor/NicheImage
cd NicheImage
python -m venv main_env
source main_env/bin/activate
pip install -e .
```
2. Select miner based on incentive distribution and subnet state at https://nicheimage.streamlit.app/
3. Setup prequisites if needed
- For StickerMaker, find the guide [here](comfyui_category.md)
- For Gemma7b, find the guide [here](vllm_category.md)
- For GoJourney, register [here](https://www.goapi.ai/midjourney-api) and get the `GOJOURNEY_API_KEY`

4. Start the endpoint

**For Image Generation Category**
```bash
source main_env/bin/activate
GOJOURNEY_API_KEY=<your-gojourney-api-key> \ # set if you use GoJourney model
PROCESS_MODE=<your-task-type> \ # set if you use GoJourney model
pm2 start python --name "miner_endpoint" -- -m services.miner_endpoint.app \
--model_name <selected-model-name> \
--num_replicas X --num_gpus Y \ # num_gpus * num_replicas = your_total_gpus_count
--port 10006 # default port
```

**For Text Generation Category**
```bash
source main_env/bin/activate
HF_TOKEN=<your-huggingface-token> \
pm2 start python --name "miner_endpoint" -- -m services.miner_endpoint.text_app --model_name <selected-model-name> --num_replicas X --num_gpus Y \
--port 10006 # default port
```

5. Start miner
```bash
pm2 start python --name "miner" \
-- \
-m neurons.miner.miner \
--netuid 23 \
--wallet.name <wallet_name> --wallet.hotkey <wallet_hotkey> \
--subtensor.network <network> \ # default is finney
--axon.port <your_public_port> \
--generate_endpoint http://127.0.0.1:10006/generate \ # change if you use different port or host
--info_endpoint http://127.0.0.1:10006/info \ # change if you use different port or host
--miner.total_volume <your-generation-volume> # default is 40. Change based on your model timeout value and GPU capacity
--miner.max_concurrent_requests <your-max-concurrent-requests> # default is 4. Change based on your model timeout value and GPU capacity
```
