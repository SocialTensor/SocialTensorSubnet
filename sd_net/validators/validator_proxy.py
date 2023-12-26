import asyncio

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from typing import Dict, Union
from concurrent.futures import ThreadPoolExecutor

import os
from sd_net.protocol import ImageGenerating
import uvicorn

class ValidatorProxy():
    def __init__(self, metagraph, dendrite, port, authentication_tokens = [], approved_urls = ""):

        self.metagraph = metagraph
        self.dendrite = dendrite
        self.port = port
        self.authentication_tokens = authentication_tokens


        print(self.port)
        print(type(self.port))
        
        self.app = FastAPI()
        self.app.add_api_route("/validator_proxy", self.forward, methods=["POST"], dependencies=[Depends(self.get_self)])

        self.start_server()


    def start_server(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.executor.submit(uvicorn.run, self.app, host="0.0.0.0", port=self.port)

    def authenticate_token(self, authorization):
        try:
            scheme, token = authorization.split()
            if scheme.lower() != "bearer":
                raise HTTPException(status_code=401, detail="Invalid authentication scheme")

            # Simple token validation (replace this with your actual validation logic)
            if token not in self.authentication_tokens:
                print("Only available tokens are:", self.authentication_tokens, flush=True)
                raise HTTPException(status_code=401, detail="Invalid token")

            return token
        except Exception as e:
            print("Exception occured in authenticating token", e, flush=True)
            raise HTTPException(status_code=401, detail="Error getting authentication token")
        

    async def forward(self, data: dict={}):

        self.authenticate_token(data["Authorization"])

        try:
            print("Received a request!",flush=True)
            payload = data.get("payload")
            uid = int(data.get("UID"))
            synapse = ImageGenerating(**payload)

            print('** IN:', uid, flush=True)

            uid_to_axon = dict(zip([int(uid) for uid in self.metagraph.uids],  self.metagraph.axons))
            axon = uid_to_axon[int(uid)]
            task = asyncio.create_task(self.dendrite.forward([axon], synapse, deserialize=True))
            await asyncio.gather(task)
            result = task.result()
            return JSONResponse(content={"status": "success", "result":result})
        except Exception as e:
            print("Exception occured in proxy forward", e, flush=True)
            raise HTTPException(status_code=400, detail=str(e))


    async def get_self(self):
        return self
