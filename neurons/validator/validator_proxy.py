from fastapi import FastAPI, HTTPException, Depends
from concurrent.futures import ThreadPoolExecutor
import requests
import torch
from image_generation_subnet.protocol import ImageGenerating
import uvicorn
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature
import bittensor as bt
import base64
import image_generation_subnet
import random
import asyncio


class ValidatorProxy:
    def __init__(
        self,
        validator,
    ):
        self.validator = validator
        self.verify_credentials = self.get_credentials()
        self.miner_request_counter = {}
        self.dendrite = bt.dendrite(wallet=validator.wallet)
        self.app = FastAPI()
        self.app.add_api_route(
            "/validator_proxy",
            self.forward,
            methods=["POST"],
            dependencies=[Depends(self.get_self)],
        )
        self.loop = asyncio.get_event_loop()

        self.start_server()

    def get_credentials(self):
        response = requests.post(
            self.validator.config.proxy.market_registering_url,
            json={
                "uid": int(self.validator.uid),
                "generate_endpoint": f"{self.validator.config.proxy.public_ip}:{self.validator.config.proxy.port}/validator_proxy",
            },
        )
        if response.status_code != 200:
            raise Exception("Error getting credentials from market api")
        response = response.json()
        message = response["message"]
        signature = response["signature"]
        signature = base64.b64decode(signature)

        def verify_credentials(public_key_bytes):
            public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
            try:
                public_key.verify(signature, message.encode("utf-8"))
            except InvalidSignature:
                raise Exception("Invalid signature")

        return verify_credentials

    def start_server(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.executor.submit(
            uvicorn.run, self.app, host="0.0.0.0", port=self.validator.config.proxy.port
        )

    def authenticate_token(self, public_key_bytes):
        public_key_bytes = base64.b64decode(public_key_bytes)
        try:
            self.verify_credentials(public_key_bytes)
            bt.logging.info("Successfully authenticated token")
            return public_key_bytes
        except Exception as e:
            print("Exception occured in authenticating token", e, flush=True)
            raise HTTPException(
                status_code=401, detail="Error getting authentication token"
            )

    def random_check(self, uid, synapse, responses, incentive_weight, checking_url):
        if random.random() < 0.01:
            bt.logging.info(f"Random check for miner {uid}")
            rewards = image_generation_subnet.validator.get_reward(
                checking_url,
                responses=responses,
                synapse=synapse,
            )
            rewards = torch.FloatTensor(rewards)
            rewards = rewards * incentive_weight
            bt.logging.info(f"Scored responses: {rewards}")
            uids = [int(uid)]
            self.validator.update_scores(rewards, uids)
            return rewards[0] > 0
        else:
            bt.logging.info("Not doing random check")
            return True

    async def forward(self, data: dict = {}):
        self.authenticate_token(data["authorization"])

        try:
            bt.logging.info("Received a request!")
            payload = data.get("payload")
            synapse = ImageGenerating(**payload)
            synapse.pipeline_params.update(
                self.validator.supporting_models[synapse.model_name]["inference_params"]
            )
            model_name = synapse.model_name
            supporting_models = self.validator.supporting_models
            scores = self.validator.scores
            metagraph = self.validator.metagraph

            available_uids = supporting_models[model_name]["uids"]
            checking_url = supporting_models[model_name]["checking_url"]
            incentive_weight = supporting_models[model_name]["incentive_weight"]

            bt.logging.info(f"Available uids: {available_uids}")
            bt.logging.info("Current scores", scores)
            if len(scores) == 0:
                scores = torch.zeros(len(metagraph.uids))
            miner_indexes = torch.multinomial(
                scores[available_uids] + 1e-6, num_samples=len(available_uids)
            )

            is_valid_response = False
            for miner_uid_index in miner_indexes:
                bt.logging.info(f"Selected miner index: {miner_uid_index}")
                miner_uid = available_uids[miner_uid_index]
                bt.logging.info(f"Selected miner uid: {miner_uid}")
                bt.logging.info(
                    f"Forwarding request to miner {miner_uid} with score {scores[miner_uid]}"
                )
                axon = metagraph.axons[miner_uid]
                bt.logging.info(f"Sending request to axon: {axon}")
                task = asyncio.create_task(
                    self.dendrite.forward(
                        [axon],
                        synapse,
                        deserialize=False,
                        timeout=60,
                    )
                )
                await asyncio.gather(task)
                responses = task.result()
                bt.logging.info(f"Received responses")
                if self.random_check(
                    miner_uid,
                    synapse,
                    responses,
                    incentive_weight,
                    checking_url,
                ):
                    is_valid_response = True
                    bt.logging.info("Checked OK")
                    break
            if not is_valid_response:
                raise Exception("No valid response")
            if miner_uid not in self.miner_request_counter:
                self.miner_request_counter[miner_uid] = 0
            self.miner_request_counter[miner_uid] += 1
            return responses[0].deserialize()
        except Exception as e:
            print("Exception occured in proxy forward", e, flush=True)
            raise HTTPException(status_code=400, detail=str(e))

    async def get_self(self):
        return self
