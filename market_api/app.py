from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import argparse
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


class Data(BaseModel):
    prompt: str = "an image of"
    seed: int = 0
    max_length: int = 77
    additional_params: dict = {}


app = FastAPI()

PRIVATE_KEY = Ed25519PrivateKey.generate()
PUBLIC_KEY = PRIVATE_KEY.public_key()
MESSAGE = "image-generating-subnet"
SIGNATURE = PRIVATE_KEY.sign(MESSAGE.encode("utf-8"))


@app.get("/get_credentials")
async def get_rewards():
    return {
        "message": MESSAGE,
        "signature": SIGNATURE,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10009)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
