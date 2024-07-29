from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import uvicorn
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature
import bittensor as bt
import base64
import logicnet
import random
import asyncio
import traceback
import httpx
import threading


class OrganicRequest(BaseModel):
    authorization: str
    synapse_request: logicnet.protocol.LogicRequest = None
    re_check: bool = False


class ValidatorProxy:
    """
    TODO. NOT IMPLEMENTED FOR LOGICNET
    """

    def __init__(
        self,
        validator,
    ):
        self.validator = validator
        self.get_credentials()
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
        if self.validator.config.proxy.port:
            self.start_server()

    def get_credentials(self):
        with httpx.Client(timeout=httpx.Timeout(30)) as client:
            response = client.post(
                f"{self.validator.config.proxy.proxy_client_url}/get_credentials",
                json={
                    "postfix": (
                        f":{self.validator.config.proxy.port}/validator_proxy"
                        if self.validator.config.proxy.port
                        else ""
                    ),
                    "uid": self.validator.uid,
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

    def organic_reward(self, synapse, response, uid, rewarder, timeout):
        if callable(rewarder):
            uids, rewards = rewarder([uid], [response], synapse)
        else:
            raise Exception("Rewarder not supported !!")
        bt.logging.info(
            f"Proxy: Updating scores of miners {uids} with rewards {rewards}"
        )

        # Scale Reward based on Miner Volume
        for i, uid in enumerate(uids):
            if rewards[i] > 0:
                rewards[i] = rewards[i] * (
                    0.9
                    + 0.1 * self.validator.miner_manager.all_uids_info[uid].reward_scale
                )

        bt.logging.info(f"Scored responses: {rewards}")

        self.validator.miner_manager.update_scores(uids, rewards)

    async def forward(self, data: OrganicRequest):
        self.authenticate_token(data.authorization)
        synapse = logicnet.protocol.LogicSynapse(**data.synapse_request.dict())
        if data.re_check:
            bt.logging.info("Rechecking validators")
            self.get_credentials()
            return {"message": "done"}
        bt.logging.info("Received an organic request!")

        category = synapse.category
        category_config = self.validator.categories[category]
        if not synapse.logic_question:
            synapse.logic_question = synapse.raw_logic_question

        timeout = category_config["timeout"]
        rewarder = category_config["rewarder"]

        metagraph = self.validator.metagraph

        output = None
        for uid, should_reward in self.validator.query_queue.get_query_for_proxy(
            category
        ):
            should_reward = (
                random.random() < self.validator.config.proxy.checking_probability
                or should_reward
            )
            bt.logging.info(
                f"Forwarding request to miner {uid} with recent scores: {self.validator.miner_manager.all_uids_info[uid].scores}"
            )
            axon = metagraph.axons[uid]
            bt.logging.info(f"Sending request to axon: {axon}")
            responses = await self.dendrite.forward(
                [axon], synapse, deserialize=False, timeout=timeout, run_async=True
            )
            response = responses[0]
            bt.logging.info(
                f"Received response from miner {uid}, status: {response.is_success}"
            )
            if should_reward:
                reward_thread = threading.Thread(
                    target=self.organic_reward,
                    args=(synapse, response, uid, rewarder, timeout),
                )
                reward_thread.start()

            if response.is_success:
                output = response
                break
            else:
                continue
        if output:
            response = output.deserialize_response()
            return response
        else:
            return HTTPException(status_code=500, detail="No valid response received")

    async def get_self(self):
        return self
