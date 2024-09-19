import argparse
from fastapi import FastAPI
import uvicorn
from openai import OpenAI
import httpx
import yaml

MODEL_CONFIG = yaml.load(
    open("generation_models/configs/model_config.yaml"), yaml.FullLoader
)

# Filter out the model names that are not VLM
MODEL_CONFIG = {
    k: v
    for k, v in MODEL_CONFIG.items()
    if "visual_question_answering" in v["params"]["supporting_pipelines"]
}


def get_args():
    parser = argparse.ArgumentParser(
        description="Run the multi modal endpoint service"
    )
    parser.add_argument(
        "--port", type=int, default=10006, help="Port to run the service on"
    )
    parser.add_argument(
        "--bind_ip", type=str, default="0.0.0.0", help="IP address to run the service on"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=MODEL_CONFIG.keys(),
        default="Pixtral_12b",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--vllm_url",
        type=str,
        default="http://localhost:8000/v1",
        help="URL of the VLLM service",
    )
    parser.add_argument(
        "--vllm_api_key",
        type=str,
        default="apikey",
        help="api key of the VLLM service",
    )

    args = parser.parse_args()
    return args


class MultiModalGenerating:
    def __init__(self, args):
        self.app = FastAPI()
        self.repo_id = MODEL_CONFIG[args.model_name]["repo_id"]
        self.args = args
        self.client = OpenAI(
            base_url=self.args.vllm_url,
            api_key=self.args.vllm_api_key,
        )
        
        self.app.add_api_route("/generate", self.generate, methods=["POST"])
        self.app.add_api_route("/info", self.info, methods=["GET"])

    async def generate(self, data: dict):
        request = {
            "model": self.repo_id,
            "logprobs": True,
            "top_logprobs": 10
        }
        request.update(data.get("pipeline_params", {}))
        message_content = [
            {
                "type": "text",
                "text": data["prompt"]
            }
        ]

        if data.get("image_url"):
                message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": data["image_url"]
                },
            })

        conversation = [
            {
                "role": "user", 
                "content": message_content
            },
        ]
        request["messages"] = conversation 
        completion =self.client.chat.completions.create(**request)
        completion = MultiModalGenerating.convert_chat_completion_ressponse_to_completion_response(completion)

        return {
            "prompt_output": completion,
        }

    async def info(self):
        return {
            "model_name": args.model_name,
        }

    @staticmethod
    def convert_chat_completion_ressponse_to_completion_response(chat_completion):
        choices = []
        for ch in chat_completion.choices:
            text_offset, offset, token_logprobs, tokens, top_logprobs = [], 0, [], [], []
            for cnt in ch.logprobs.content:
                text_offset.append(offset)
                offset += len(cnt.token)
                token_logprobs.append(cnt.logprob)
                tokens.append(cnt.token)
                top_logprobs.append({x.token: x.logprob for x in cnt.top_logprobs})
            choice = {
                "index": ch.index,
                "text": ch.message.content,
                "logprobs": {
                    "text_offset": text_offset,
                    "token_logprobs": token_logprobs,
                    "tokens": tokens,
                    "top_logprobs": top_logprobs
                },
                
            }
            choices.append(choice)
        chat_completion.choices = choices

        completion = chat_completion.__dict__
        return completion

if __name__ == "__main__":
    args = get_args()
    app = MultiModalGenerating(args)

    uvicorn.run(app.app, host=args.bind_ip, port=args.port)
