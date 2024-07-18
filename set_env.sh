MINER_BASE_URL=http://localhost:8000/v1
MINER_MODEL="Qwen/Qwen2-7B-Instruct"
MINER_KEY="x"
REWARD_BASE_URL=http://localhost:8000/v1
REWARD_MODEL="Qwen/Qwen2-7B-Instruct"
REWARD_KEY="x"
CHALLENGE_BASE_URL=http://localhost:8000/v1
CHALLENGE_MODEL="Qwen/Qwen2-7B-Instruct"
CHALLENGE_KEY="x"

# generate .env file
echo "MINER_BASE_URL=$MINER_BASE_URL" > .env
echo "MINER_MODEL=$MINER_MODEL" >> .env
echo "MINER_KEY=$MINER_KEY" >> .env
echo "REWARD_BASE_URL=$REWARD_BASE_URL" >> .env
echo "REWARD_MODEL=$REWARD_MODEL" >> .env
echo "REWARD_KEY=$REWARD_KEY" >> .env
echo "CHALLENGE_BASE_URL=$CHALLENGE_BASE_URL" >> .env
echo "CHALLENGE_MODEL=$CHALLENGE_MODEL" >> .env
echo "CHALLENGE_KEY=$CHALLENGE_KEY" >> .env
