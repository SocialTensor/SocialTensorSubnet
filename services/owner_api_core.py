from fastapi import Response
from starlette.requests import Request
import bittensor as bt
import time
from slowapi import Limiter
from pprint import pprint


def get_forwarded_for(request: Request):
    return request.headers.get("x-forwarded-for")


limiter = Limiter(key_func=get_forwarded_for)


@limiter.limit("120/minute")
async def filter_allowed_ips(self, request: Request, call_next):
    if self.args.disable_secure:
        response = await call_next(request)
        return response
    forwarded_for = request.headers.get("x-forwarded-for")
    print("Forwarded for:", forwarded_for, flush=True)
    if (
        (forwarded_for not in self.allowed_ips)
        and (request.client.host != "127.0.0.1")
        and self.allowed_ips
    ):
        print("Blocking an unallowed ip:", forwarded_for, flush=True)
        return Response(
            content="You do not have permission to access this resource",
            status_code=403,
        )
    print("Allow an ip:", forwarded_for, flush=True)
    response = await call_next(request)
    return response


def define_allowed_ips(self, url, netuid, min_stake):
    while True:
        try:
            state = {}
            all_allowed_ips = []
            subtensor = bt.subtensor(url)
            metagraph = subtensor.metagraph(netuid)
            for uid in range(len(metagraph.total_stake)):
                if metagraph.total_stake[uid] > min_stake:
                    all_allowed_ips.append(metagraph.axons[uid].ip)
                    state[uid] = {
                        "stake": metagraph.total_stake[uid].item(),
                        "ip": metagraph.axons[uid].ip,
                    }
            self.allowed_ips = all_allowed_ips
            # sort by stake
            state = dict(
                sorted(state.items(), key=lambda item: item[1]["stake"], reverse=True)
            )
            print("Updated allowed ips", flush=True)
            pprint(state)
        except Exception as e:
            print("Exception while updating allowed ips", str(e), flush=True)
        time.sleep(60)