import asyncio
from fastapi import FastAPI, HTTPException, Depends
from concurrent.futures import ThreadPoolExecutor
import requests
import torch
from image_generation_subnet.protocol import ImageGenerating
import uvicorn
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature
import bittensor as bt


class ValidatorProxy:
    def __init__(self, metagraph, dendrite, port, market_url, supporting_models):
        self.metagraph = metagraph
        self.dendrite = dendrite
        self.port = port
        self.market_url = market_url
        self.verify_credentials = self.get_credentials()
        self.supporting_models = supporting_models
        self.miner_request_counter = {}

        self.app = FastAPI()
        self.app.add_api_route(
            "/validator_proxy",
            self.forward,
            methods=["POST"],
            dependencies=[Depends(self.get_self)],
        )

        self.start_server()

    def get_credentials(self):
        response = requests.get(self.market_url)
        if response.status_code != 200:
            raise Exception("Error getting credentials from market api")
        response = response.json()
        message = response["message"]
        signature = response["signature"]

        def verify_credentials(public_key_bytes):
            public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
            try:
                public_key.verify(signature, message.encode("utf-8"))
            except InvalidSignature:
                raise Exception("Invalid signature")

        return verify_credentials

    def start_server(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.executor.submit(uvicorn.run, self.app, host="0.0.0.0", port=self.port)

    def authenticate_token(self, public_key_bytes):
        try:
            self.verify_credentials(public_key_bytes)
            return public_key_bytes
        except Exception as e:
            print("Exception occured in authenticating token", e, flush=True)
            raise HTTPException(
                status_code=401, detail="Error getting authentication token"
            )

    async def forward(self, data: dict = {}):
        self.authenticate_token(data["authorization"])

        try:
            bt.logging.info("Received a request!")
            payload = data.get("payload")
            synapse = ImageGenerating(**payload)
            model_name = synapse.model_name

            available_uids = self.supporting_models[model_name]["uids"]
            weights = self.metagraph.weights[available_uids]
            miner_uid = torch.multinomial(weights, 1).item()
            axon = self.metagraph.axons[miner_uid]
            task = asyncio.create_task(
                self.dendrite.forward([axon], synapse, deserialize=True)
            )
            await asyncio.gather(task)
            result = task.result()
            if miner_uid not in self.miner_request_counter:
                self.miner_request_counter[miner_uid] = 0
            self.miner_request_counter[miner_uid] += 1
            return result
        except Exception as e:
            print("Exception occured in proxy forward", e, flush=True)
            raise HTTPException(status_code=400, detail=str(e))

    async def get_self(self):
        return self
