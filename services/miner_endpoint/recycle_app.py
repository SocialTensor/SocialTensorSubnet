import argparse

import uvicorn
from fastapi import FastAPI


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10006)
    parser.add_argument(
        "--bind_ip",
        type=str,
        default="0.0.0.0",
        help="IP address to run the service on",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Recycle",
    )
    args = parser.parse_args()
    return args


args = get_args()

class MinerEndpoint:
    def __init__(self):
        self.app = FastAPI()
        self.app.add_api_route("/info", self.info, methods=["GET"])

    async def info(self):
        return {
            "model_name": args.model_name,
        }
    
if __name__ == "__main__":
    app = MinerEndpoint()
    uvicorn.run(
        app.app,
        host=args.bind_ip,
        port=args.port,
    )