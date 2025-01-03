import os
import sys
sys.path.append("../")
from logicnet.validator import LogicChallenger
from logicnet.protocol import LogicSynapse
from dotenv import load_dotenv
load_dotenv()

synapse = LogicSynapse()

MODEL = os.getenv("MINER_MODEL", "gpt-4o-mini")
BASE_URL = os.getenv("MINER_BASE_URL", "https://api.openai.com/v1")
KEY = os.getenv("MINER_KEY")
DATASET_WEIGHT = "20,20,20,20,20,20,20"
print(MODEL, BASE_URL, KEY)

model_rotation_pool = {
    "gpt-4o": [BASE_URL, KEY, "gpt-4o-mini"],
}
challenger = LogicChallenger(
    model_rotation_pool=model_rotation_pool,
    dataset_weight=DATASET_WEIGHT,
)


for _ in range(20):
    challenger(synapse)
    print(synapse)
    print()
