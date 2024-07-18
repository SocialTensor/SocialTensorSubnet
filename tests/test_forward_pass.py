import logicnet
import openai
import os
import asyncio
import copy
import time
from dotenv import load_dotenv

load_dotenv(override=True)


def get_or_create_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


MODEL = os.getenv("MINER_MODEL", "gpt-3.5-turbo")
BASE_URL = os.getenv("MINER_BASE_URL", "https://api.openai.com/v1")
KEY = os.getenv("MINER_KEY")

print(MODEL, BASE_URL)

synapse = logicnet.protocol.LogicSynapse()

challenger = logicnet.validator.LogicChallenger()
rewarder = logicnet.validator.LogicRewarder()

synapse = challenger(synapse)
synapse.timeout = 12

base_synapse = copy.deepcopy(synapse)

synapse = synapse.miner_synapse()

client = openai.AsyncOpenAI(base_url=BASE_URL, api_key=KEY)

start = time.time()
_solver = logicnet.miner.solve(synapse, client)

loop = get_or_create_loop()
synapse = loop.run_until_complete(_solver)
duration = time.time() - start
synapse.dendrite.process_time = duration

rewards = rewarder([0], [synapse], base_synapse)
