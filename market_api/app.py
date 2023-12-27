from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import argparse
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
import bittensor as bt
import random
import requests
import io
import base64
from PIL import Image


class Prompt(BaseModel):
    prompt: str = "an image of"
    model_name: str = "RealisticVision"


class ValidatorInfo(BaseModel):
    uid: int
    generate_endpoint: str
    counter: int = 0


def base64_to_pil_image(base64_image: str) -> Image.Image:
    image_stream = io.BytesIO(base64.b64decode(base64_image))
    image = Image.open(image_stream)

    return image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10009)
    parser.add_argument("--chain_endpoint", type=str, default="ws://20.243.203.20:9946")
    args = parser.parse_args()
    return args


app = FastAPI()

PRIVATE_KEY = Ed25519PrivateKey.generate()
PUBLIC_KEY = PRIVATE_KEY.public_key()
PUBLIC_KEY_BYTES = PUBLIC_KEY.public_bytes(
    encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
)
MESSAGE = "image-generating-subnet"
SIGNATURE = base64.b64encode(PRIVATE_KEY.sign(MESSAGE.encode("utf-8")))
AVAILABLE_VALIDATORS = []
ARGS = parse_args()


@app.post("/get_credentials")
async def get_credentials(validator_info: ValidatorInfo):
    print(validator_info, flush=True)
    AVAILABLE_VALIDATORS.append(validator_info)
    return {
        "message": MESSAGE,
        "signature": SIGNATURE,
    }


@app.post("/generate")
async def generate(prompt: Prompt):
    subtensor = bt.subtensor(ARGS.chain_endpoint)
    metagraph = subtensor.metagraph(1)
    stakes = [
        metagraph.total_stake[validator.uid] for validator in AVAILABLE_VALIDATORS
    ]
    validator = random.choices(AVAILABLE_VALIDATORS, weights=stakes)[0]
    request_dict = {
        "payload": {
            "prompt": prompt.prompt,
            "model_name": prompt.model_name,
        },
        "model_name": prompt.model_name,
        "authorization": base64.b64encode(PUBLIC_KEY_BYTES).decode("utf-8"),
    }
    response = requests.post(validator.generate_endpoint, json=request_dict)
    if response.status_code != 200:
        raise Exception("Error generating image")
    else:
        validator.counter += 1
    response = response.json()
    return response


uvicorn.run(app, host="0.0.0.0", port=ARGS.port)
