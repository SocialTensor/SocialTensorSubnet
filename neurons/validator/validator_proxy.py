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
import os
import random
import asyncio
from image_generation_subnet.validator.proxy import ProxyCounter


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
        self.proxy_counter = ProxyCounter(
            os.path.join(self.validator.config.neuron.full_path, "proxy_counter.json")
        )
        self.start_server()

    def get_credentials(self):
        response = requests.post(
            f"{self.validator.config.proxy.proxy_client_url}/get_credentials",
            json={
                "postfix": f":{self.validator.config.proxy.port}/validator_proxy",
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

    def random_check(self, uid, synapse, response, checking_url):
        if random.random() < self.validator.config.proxy.checking_probability:
            bt.logging.info(f"Random check for miner {uid}")
            rewards = image_generation_subnet.validator.get_reward(
                checking_url,
                responses=[response],
                synapse=synapse,
            )
            if rewards is None:
                return False
            self.validator.all_uids_info[str(uid)]["scores"].append(rewards[0])
            bt.logging.info(f"Scored responses: {rewards}")
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

            available_uids = [
                int(uid)
                for uid in self.validator.all_uids_info.keys()
                if self.validator.all_uids_info[uid]["model_name"] == model_name
            ]
            checking_url = supporting_models[model_name]["checking_url"]
            incentive_weight = supporting_models[model_name]["incentive_weight"]

            bt.logging.info(f"Available uids: {available_uids}")
            bt.logging.info("Current scores", scores)
            if len(scores) == 0:
                scores = torch.zeros(len(metagraph.uids))
            available_uids = [
                uid
                for uid in available_uids
                if scores[uid] > self.validator.config.proxy.miner_score_threshold
            ]
            if len(available_uids) == 0:
                raise Exception("No miners meet the score threshold")
            scores = scores[available_uids]
            miner_indexes = torch.multinomial(
                scores + 1e-6, num_samples=len(available_uids)
            )

            is_valid_response = False
            for miner_uid_index in miner_indexes:
                bt.logging.info(f"Selected miner index: {miner_uid_index}")
                miner_uid = available_uids[miner_uid_index]
                bt.logging.info(f"Selected miner uid: {miner_uid}")
                bt.logging.info(
                    f"Forwarding request to miner {miner_uid} with score {scores[miner_uid_index]}"
                )
                axon = metagraph.axons[miner_uid]
                bt.logging.info(f"Sending request to axon: {axon}")
                task = asyncio.create_task(
                    self.dendrite.forward(
                        [axon],
                        synapse,
                        deserialize=False,
                        timeout=self.validator.supporting_models[model_name]["timeout"],
                    )
                )
                await asyncio.gather(task)
                response = task.result()[0]
                bt.logging.info(f"Received responses")
                if self.random_check(
                    miner_uid,
                    synapse,
                    response,
                    checking_url,
                ):
                    is_valid_response = True
                    bt.logging.info("Checked OK")
                    break
            if not is_valid_response:
                raise Exception("No valid response")
            try:
                self.proxy_counter.update(is_success=is_valid_response)
            except Exception as e:
                print("Exception occured in updating proxy counter", e, flush=True)
            return response.deserialize()
        except Exception as e:
            print("Exception occured in proxy forward", e, flush=True)
            raise HTTPException(status_code=400, detail=str(e))

    async def get_self(self):
        return self
