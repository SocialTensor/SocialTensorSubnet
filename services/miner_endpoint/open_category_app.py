import argparse
import litserve as ls
from generation_models import OpenModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_gpus", default=1, type=int
    )
    parser.add_argument(
        "--port", default=10006, type=int
    )
    parser.add_argument(
        "--model_id", default=""
    )
    parser.add_argument(
        "--category", choices=["OpenGeneral", "OpenLandscape"], default="OpenGeneral"
    )

    args = parser.parse_args()

    core = OpenModel(args.category, args.model_id)

    server = ls.LitServer(core, accelerator="auto", max_batch_size=1, devices=args.num_gpus, api_path="/generate")
    server.app.add_api_route(
        "/info",
        core.get_info,
        methods=["GET"]
    )
    server.run(port=args.port)