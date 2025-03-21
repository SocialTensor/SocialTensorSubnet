from .base_model import BaseModel
import os
import httpx
import bittensor as bt

API_KEY = os.getenv("GOJOURNEY_API_KEY")
PROCESS_MODE = os.getenv("PROCESS_MODE", "relax")


class NicheGoJourney(BaseModel):
    def __init__(self, *args, **kwargs):
        assert API_KEY, "GOJOURNEY_API_KEY is not set"
        self.inference_function = self.load_model(*args, **kwargs)

    def load_model(self, *args, **kwargs):
        imagine_inference_function = self.load_imagine(*args, **kwargs)
        return imagine_inference_function

    def __call__(self, *args, **kwargs):
        return self.inference_function(*args, **kwargs)

    def load_imagine(self, *args, **kwargs):
        imagine_endpoint = "https://api.goapi.ai/api/v1/task"
        fetch_endpoint = "https://api.goapi.ai/api/v1/task/{task_id}"
        headers = {
            "x-api-key": API_KEY,
            "Content-Type": "application/json"
        }

        def inference_function(*args, **kwargs):
            data = {
                "model": "midjourney",
                "task_type": "imagine",
                "input": {
                    "prompt": kwargs["prompt"],
                    "process_mode": kwargs["pipeline_params"].get("process_mode", PROCESS_MODE),
                }
                
            }
            with httpx.Client() as client:
                imagine_response = client.post(
                    imagine_endpoint, headers=headers, json=data, timeout=32
                )
                imagine_response = imagine_response.json()
                bt.logging.info(imagine_response)
                task_id = imagine_response["data"]["task_id"]
            return {"task_id": task_id}

        return inference_function
