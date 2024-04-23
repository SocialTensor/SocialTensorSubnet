import argparse
from fastapi import FastAPI
import uvicorn
from transformers import AutoTokenizer
import httpx
import yaml

MODEL_CONFIG = yaml.load(
    open("generation_models/configs/model_config.yaml"), yaml.FullLoader
)

# Filter out the model names that are not LLM
MODEL_CONFIG = {
    k: v
    for k, v in MODEL_CONFIG.items()
    if "text_generation" in v["params"]["supporting_pipelines"]
}


def get_args():
    parser = argparse.ArgumentParser(
        description="Run the LLM Prompt Generating service"
    )
    parser.add_argument(
        "--port", type=int, default=10006, help="Port to run the service on"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=MODEL_CONFIG.keys(),
        default="Gemma7b",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--vllm_url",
        type=str,
        default="http://localhost:8000",
        help="URL of the VLLM service",
    )
    args = parser.parse_args()
    return args


class LLMPromptGenerating:
    def __init__(self, args):
        self.app = FastAPI()
        self.repo_id = MODEL_CONFIG[args.model_name]["repo_id"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id)
        self.args = args
        self.app.add_api_route("/generate", self.generate, methods=["POST"])
        self.app.add_api_route("/info", self.info, methods=["GET"])

    async def generate(self, data: dict):
        data["model"] = self.repo_id
        async with httpx.AsyncClient() as httpx_client:
            response = await httpx_client.post(
                f"{self.args.vllm_url}/v1/completions",
                json=data,
                timeout=data["timeout"],
            )
        response.raise_for_status()
        response: dict = response.json()
        return {
            "prompt_output": response,
        }

    async def info(self):
        return {
            "model_name": args.model_name,
        }


if __name__ == "__main__":
    args = get_args()
    llm_prompt_generating = LLMPromptGenerating(args)

    uvicorn.run(llm_prompt_generating.app, host="0.0.0.0", port=args.port)
