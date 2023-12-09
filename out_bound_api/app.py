from fastapi import FastAPI
import bittensor as bt
from typing import List
import random
from pydantic import BaseModel
import argparse
import uvicorn

app = FastAPI()

@app.get("/get_allowed_ip_list")
async def get_rewards():
    all_allowed_ips = []
    subtensor = bt.subtensor("ws://127.0.0.1:9946")
    metagraph = subtensor.metagraph(args.netuid)
    for uid in range(len(metagraph.total_stake)):
        if metagraph.total_stake[uid] > args.min_stake:
            all_allowed_ips.append(metagraph.axons[uid].ip)
    return {"allowed_ips": all_allowed_ips}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10005)
    parser.add_argument("--subnet_url", type=str, default="ws://127.0.0.1:9946")
    parser.add_argument("--netuid", type=int, default=1)
    parser.add_argument("--min_stake", type=int, default=100)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
