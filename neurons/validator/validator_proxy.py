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
import requests


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

    def get_credentials(self):
        response = requests.post(
            f"{self.validator.config.proxy.proxy_client_url}/get_credentials",
            json={
                "postfix": (
                    f":{self.validator.config.proxy.port}/validator_proxy"
                    if self.validator.config.proxy.port
                    else ""
                ),
                "uid": self.validator.uid,
                "all_uid_info": self.validator.miner_manager.all_uids_info,
                "SHA": "WED6MAR",
            },
            timeout=30,
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

    def organic_reward(self, uid, synapse, response, url):
        if not len(response.image) and isinstance(url, str):
            bt.logging.info(f"Empty image for miner {uid}")
            self.validator.miner_manager.all_uids_info[uid]["scores"].append(0)
            return False
        if (
            random.random() < self.validator.config.proxy.checking_probability
            or callable(url)
        ):
            bt.logging.info(f"Rewarding an organic request for miner {uid}")
            if callable(url):
                rewards = url(synapse, response, [uid])
            else:
                rewards = image_generation_subnet.validator.get_reward(
                    url, synapse, [response], [uid]
                )
            if rewards is None:
                return False
            self.validator.miner_manager.update_scores([uid], rewards)
            bt.logging.info(f"Organic rewards: {rewards}")
            return rewards[0] > 0
        else:
            bt.logging.info("Not doing organic reward")
            return True

    async def forward(self, data: dict = {}):
        self.authenticate_token(data["authorization"])
        payload = data.get("payload")
        miner_uid = data.get("miner_uid", -1)
        if "recheck" in payload:
            bt.logging.info("Rechecking validators")
            self.verify_credentials = self.get_credentials()
            return {"message": "Rechecked"}
        try:
            bt.logging.info("Received a request!")
            if "seed" not in payload:
                payload["seed"] = random.randint(0, 1e9)
            model_name = payload["model_name"]
            synapse = ImageGenerating(**payload)
            synapse.limit_params()

            # Override default pipeline params
            for k, v in self.validator.nicheimage_catalogue[synapse.model_name][
                "inference_params"
            ].items():
                if k not in synapse.pipeline_params:
                    synapse.pipeline_params[k] = v
            timeout = self.validator.nicheimage_catalogue[model_name]["timeout"] * 2
            metagraph = self.validator.metagraph
            reward_url = self.validator.nicheimage_catalogue[model_name]["reward_url"]

            specific_weights = self.validator.miner_manager.get_model_specific_weights(
                model_name, normalize=False
            )

            if miner_uid >= 0:
                if specific_weights[miner_uid] == 0:
                    raise Exception("Selected miner score is 0")
                available_uids = [miner_uid]
            else:
                available_uids = [
                    i
                    for i in range(len(specific_weights))
                    if specific_weights[i]
                    > self.validator.config.proxy.miner_score_threshold
                ]

                if not available_uids:
                    bt.logging.warning(
                        "No miners meet the score threshold, selecting all non-zero miners"
                    )
                    available_uids = [
                        i
                        for i in range(len(specific_weights))
                        if specific_weights[i] > 0
                    ]
                    if not available_uids:
                        raise Exception("No miners available")
            is_valid_response = False
            random.shuffle(available_uids)
            for uid in available_uids[:5]:
                bt.logging.info(f"Selected miner uid: {uid}")
                bt.logging.info(
                    f"Forwarding request to miner {uid} with score {specific_weights[uid]}"
                )
                axon = metagraph.axons[uid]
                bt.logging.info(f"Sending request to axon: {axon}")
                task = asyncio.create_task(
                    self.dendrite.forward(
                        [axon],
                        synapse,
                        deserialize=False,
                        timeout=timeout,
                    )
                )
                await asyncio.gather(task)
                response = task.result()[0]
                bt.logging.info("Received responses")
                if self.organic_reward(
                    miner_uid,
                    synapse,
                    response,
                    reward_url,
                ):
                    is_valid_response = True
                    bt.logging.info("Checked OK")
                    break
                else:
                    bt.logging.info("Checked not OK, trying another miner")
                    continue

            if not is_valid_response:
                raise Exception("No valid response")
            try:
                self.proxy_counter.update(is_success=is_valid_response)
                self.proxy_counter.save()
            except Exception:
                print(
                    "Exception occured in updating proxy counter",
                    traceback.format_exc(),
                    flush=True,
                )
            response = response.deserialize()
            if response.get("image", ""):
                return response["image"]
            else:
                return response["response_dict"]
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def get_self(self):
        return self
