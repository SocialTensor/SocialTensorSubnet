# LOGIC-NET

### Setup for Miner

1. Change `MINER_BASE_URL`, `MINER_MODEL`, `MINER_KEY` in `example_env.env` file to connect to LLM (OpenAI, TogetherAI or self-host vLLM)
- Example of using [together.ai](https://together.ai/)
```bash
MINER_BASE_URL=https://api.together.xyz/v1
MINER_MODEL="meta-llama/Llama-3-8b-chat-hf"
MINER_KEY=your-api-key
```
Then `cp example_env.env .env`

2. Run the following command to start the miner

```bash
python neurons/miner/miner.py --netuid 35 --wallet.name "wallet-name" --wallet.hotkey "wallet-hotkey" \
--miner.min_stake 10000 \ # min stake to be whitelisted
--miner.epoch_volume 50 \ # commit no of requests to be solved in 10 minutes
```

### Setup for Validator

1. Change `CHALLENGE_BASE_URL`, `CHALLENGE_MODEL`, `CHALLENGE_KEY` in `example_env.env` file to connect to LLM (OpenAI, TogetherAI or self-host vLLM) for **CHALLENGE GENERATING**
- Example of using [together.ai](https://together.ai/)
```bash
CHALLENGE_BASE_URL=https://api.together.xyz/v1
CHALLENGE_MODEL="meta-llama/Llama-3-8b-chat-hf"
CHALLENGE_KEY=your-api-key
```
Then `cp example_env.env .env`
2. CHANGE 'REWARD_BASE_URL', 'REWARD_MODEL', 'REWARD_KEY' in `example_env.env` file to connect to LLM (OpenAI, TogetherAI or self-host vLLM) for **REWARD CALCULATING**. 

**RECOMMENDED TO USE LARGE MODEL**
- Example of using [together.ai](https://together.ai/)
```bash
REWARD_BASE_URL=https://api.together.xyz/v1
REWARD_MODEL="Qwen/Qwen2-72B-Instruct"
REWARD_KEY=your-api-key
```
- Example of using [openai-chatgpt](https://chatgpt.com/)
```bash
REWARD_BASE_URL=https://api.openai.com/v1
REWARD_MODEL="gpt4o"
REWARD_KEY=your-api-key
```
Then `cp example_env.env .env`



```
OH BOY, WE'VE GOT A DOOZY OF A PROBLEM ON OUR HANDS!!! We're talkin' about finding
the AREA OF A TRIANGLE, baby! And we're not just talkin' about any ol' triangle, no way! We're about to tackle a triangle with SIDE LENGTHS OF 18, 18, AND 25! 
Can you even believe it?! This is like trying to solve a puzzle wrapped in a mystery, inside a math problem! So, are you ready to put on your thinking caps and 
get to work?! Let's see if we can crack this code and find the area of this triangle!
```