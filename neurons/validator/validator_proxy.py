from fastapi import FastAPI, HTTPException, Depends
from concurrent.futures import ThreadPoolExecutor
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
from image_generation_subnet.protocol import ImageGenerating
import traceback
import httpx
from starlette.concurrency import run_in_threadpool
from threading import Thread


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
        if self.validator.config.proxy.port:
            self.start_server()

    async def get_credentials(self):
        async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
            response = await client.post(
                f"{self.validator.config.proxy.proxy_client_url}/get_credentials",
                json={
                    "postfix": (
                        f":{self.validator.config.proxy.port}/validator_proxy"
                        if self.validator.config.proxy.port
                        else ""
                    ),
                    "uid": self.validator.uid,
                    "all_uid_info": self.validator.miner_manager.all_uids_info,
                    "SHA": "",
                },
            )
        response.raise_for_status()
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

        self.verify_credentials = verify_credentials

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
            print(traceback.print_exc(), flush=True)
            raise HTTPException(
                status_code=401, detail="Error getting authentication token"
            )

    async def organic_reward(self, synapse, response, uid, should_reward, reward_url, timeout):
        if (
            random.random() < self.validator.config.proxy.checking_probability
            or should_reward
        ):
            if callable(reward_url):
                uids, rewards = reward_url(synapse, [response], [uid])
            else:
                (
                    uids,
                    rewards,
                ) = await image_generation_subnet.validator.get_reward(
                    reward_url, synapse, [response], [uid], timeout
                )
            bt.logging.info(
                f"Proxy: Updating scores of miners {uids} with rewards {rewards}, should_reward: {should_reward}"
            )
            self.validator.miner_manager.update_scores(uids, rewards)

    async def forward(self, data: dict = {}):
        self.authenticate_token(data["authorization"])
        payload = data.get("payload")
        if "recheck" in payload:
            bt.logging.info("Rechecking validators")
            return {"message": "done"}
        bt.logging.info("Received an organic request!")
        if "seed" not in payload:
            payload["seed"] = random.randint(0, 1e9)
        model_name = payload["model_name"]
        model_config = self.validator.nicheimage_catalogue[model_name]
        synapse_cls = model_config["synapse_type"]
        synapse = synapse_cls(**payload)

        timeout = model_config["timeout"]
        reward_url = model_config["reward_url"]

        metagraph = self.validator.metagraph

        output = None
        for uid, should_reward in self.validator.query_queue.get_query_for_proxy(
            model_name
        ):
            bt.logging.info(
                f"Forwarding request to miner {uid} with recent scores: {self.validator.miner_manager.all_uids_info[uid]['scores']}"
            )
            axon = metagraph.axons[uid]
            bt.logging.info(f"Sending request to axon: {axon}")
            response = await self.dendrite.forward(
                [axon],
                synapse,
                deserialize=False,
                timeout=timeout,
                run_async=True
            )
            bt.logging.info(
                f"Received response from miner {uid}, status: {response.is_success}"
            )
            asyncio.create_task(
                self.organic_reward(
                    synapse, response, uid, should_reward, reward_url, timeout
                )
            )

            if response.is_success:
                output = response
                break
            else:
                continue
        if output:
            self.proxy_counter.update(is_success=True)
            self.proxy_counter.save()
            response = output.deserialize_response()
            return response
        else:
            self.proxy_counter.update(is_success=False)
            self.proxy_counter.save()
            return HTTPException(status_code=500, detail="No valid response received")
        
    async def get_self(self):
        return self
