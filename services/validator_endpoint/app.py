import argparse
import uvicorn
import yaml
import asyncio
import time, json
import gc
import torch
from fastapi import FastAPI
from typing import Optional, List
from pydantic import BaseModel, Extra
from services.rays.image_generating import ModelDeployment
from services.challenge_generating.image_generating.app import ImageGenerator, TextToImagePrompt
from services.challenge_generating.prompt_generating.app import Prompt as ChallengePromptInput
from services.challenge_generating.prompt_generating.model import ChallengePrompt

MODEL_CONFIG = yaml.load(
    open("generation_models/configs/model_config.yaml"), yaml.FullLoader
)


class RequestCancelledMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Let's make a shared queue for the request messages
        queue = asyncio.Queue()

        async def message_poller(sentinel, handler_task):
            nonlocal queue
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    handler_task.cancel()
                    return sentinel  # Break the loop

                # Puts the message in the queue
                await queue.put(message)

        sentinel = object()
        handler_task = asyncio.create_task(self.app(scope, queue.get, send))
        asyncio.create_task(message_poller(sentinel, handler_task))

        try:
            return await handler_task
        except asyncio.CancelledError:
            print("Cancelling request due to disconnect")


class Prompt(BaseModel, extra=Extra.allow):
    prompt: str
    seed: int
    pipeline_type: str
    pipeline_params: Optional[dict] = {}

class PromptRequests(BaseModel):
    model_name: str
    prompts: List[Prompt]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=13300)
    parser.add_argument(
        "--bind_ip",
        type=str,
        default="0.0.0.0",
        help="IP address to run the service on",
    )
    args = parser.parse_args()
    return args


args = get_args()


class ValidatorEndpoint:
    def __init__(self):
        self.challenge_prompt = ChallengePrompt()
        self.model_handle = None
        self.model_name = None
        self.app = FastAPI()
        self.app.add_api_route("/generate", self.generate, methods=["POST"])
        self.app.add_api_route("/challenge/prompt", self.generate_challenge_prompt, methods=["POST"])
        self.app.add_api_route("/challenge/image", self.generate_challenge_image, methods=["POST"])
        self.challenge_image_model = "challenge_image_model"
        self.app.add_middleware(RequestCancelledMiddleware)

    def load_model(self, model_name):
        print(f"Loading model: {model_name}")
        start = time.time()
        model_deployment = ModelDeployment(MODEL_CONFIG[model_name])
        print(f"Load {model_name} time: {time.time() - start}")
        return model_deployment

    async def unload_model(self):
        if hasattr(self, 'model_handle'):
            del self.model_handle
        torch.cuda.empty_cache()
        gc.collect()
        

    async def generate(self, item: PromptRequests):
        model_name = item.model_name

        if model_name != self.model_name:
            asyncio.get_event_loop().run_until_complete(self.unload_model())
            self.model_handle = self.load_model(model_name)
            self.model_name = model_name
        
        outputs = []
        for prompt_data in item.prompts:
            prompt_data = prompt_data.dict()
            output = await self.model_handle.generate(prompt_data=prompt_data)
            if isinstance(output, dict):
                prompt_data["response_dict"] = output
            if isinstance(output, str):
                prompt_data["image"] = output
            outputs.append(prompt_data)
        return outputs

    async def generate_challenge_image(self, item: TextToImagePrompt):
        model_name = self.challenge_image_model
        if model_name != self.model_name:
            asyncio.get_event_loop().run_until_complete(self.unload_model())
            self.model_handle = ImageGenerator()
            self.model_name = model_name
        
        result = await self.model_handle(item.__dict__)
        return  {"conditional_image": result}
        
    def generate_challenge_prompt(self, item: ChallengePromptInput):
        data = dict(item)
        prompt = data["prompt"]
        if not prompt:
            prompt = "an image of "
        complete_prompt = self.challenge_prompt.infer_prompt([prompt], max_generation_length=77, sampling_topk=30, sampling_topp=0.9)
        return {"prompt": complete_prompt}
    
if __name__ == "__main__":
    app = ValidatorEndpoint()
    uvicorn.run(
        app.app,
        host=args.bind_ip,
        port=args.port,
    )
