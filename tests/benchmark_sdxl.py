from neurons.validator.validator import initialize_nicheimage_catalogue, Validator
from image_generation_subnet.protocol import ImageGenerating
from services.rays.image_generating import ModelDeployment
import argparse
import yaml
import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor
import random
from image_generation_subnet.utils.config import config, add_args
import bittensor as bt
import matplotlib.pyplot as plt
import numpy as np
from generation_models.utils import pil_image_to_base64
from PIL import Image

def plot_report(reports):
    for status_code, times in reports.items():
        if status_code != 200:
            continue
        mean = np.mean(times)
        std = np.std(times)
        plt.hist(times, bins=4, alpha=0.7, label=f"Status code: {status_code}")
        plt.axvline(mean, color='k', linestyle='dashed', linewidth=1)
        plt.text(mean, 0, f"Mean: {mean:.2f}", rotation=90)
        plt.axvline(mean + std, color='r', linestyle='dashed', linewidth=1)
        plt.text(mean + std, 0, f"Mean + std: {mean + std:.2f}", rotation=90)
        plt.axvline(mean - std, color='r', linestyle='dashed', linewidth=1)
        plt.text(mean - std, 0, f"Mean - std: {mean - std:.2f}", rotation=90)

    plt.xlabel("Time taken (s)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("tests/benchmark.png")

def get_catalogue():
    parser = argparse.ArgumentParser()
    add_args(Validator, parser)

    CONFIG = bt.config(parser)
    return initialize_nicheimage_catalogue(CONFIG)

CONFIG = yaml.load(open("generation_models/configs/model_config.yaml"), yaml.FullLoader)

parser = argparse.ArgumentParser(description='Benchmark SDXL')
parser.add_argument('--n_times', type=int, default=10, help='Number of times to benchmark')
parser.add_argument("--model_name", type=str, default="RealitiesEdgeXL", help="Model name")
parser.add_argument("--max_concurrent_requests", type=int, default=1, help="Number of concurrent requests")
args = parser.parse_args()

model_catalogue = get_catalogue()
MODEL_CONFIG = CONFIG[args.model_name]
pipelines = MODEL_CONFIG["params"]["supporting_pipelines"]

print(f"Model catalogue: {model_catalogue[args.model_name]}")
print(f"Supporting pipelines: {pipelines}")

synapse = ImageGenerating(model_name=args.model_name, timeout=model_catalogue[args.model_name]["timeout"])
synapse.prompt = "a cute cat"
synapse.seed = random.randint(0, 1000000)
synapse.pipeline_params = model_catalogue[args.model_name]["inference_params"]
synapse.conditional_image = pil_image_to_base64(Image.open("assets/images/image.png"))



def benchmark_sdxl(n_times, model_name, n_concurrent_requests):
    # report times, status codes
    import time
    def _post(synapse: ImageGenerating):
        timeout = synapse.timeout
        # synapse.pipeline_type = random.choice(pipelines)
        synapse.pipeline_type = "txt2img"
        print(synapse.pipeline_type)
        start = time.time()
        try:
            response = requests.post("http://localhost:10006/generate", json=synapse.deserialize_input(), timeout=timeout)
        except requests.exceptions.ReadTimeout:
            return 408, timeout
        end = time.time()
        return response.status_code, end - start
    reports = {}
    print(f"Benchmarking {model_name} with {n_times} times and {n_concurrent_requests} concurrent requests")
    print("Starting benchmark...")
    for _ in tqdm.tqdm(range(n_times)):
        with ThreadPoolExecutor(max_workers=n_concurrent_requests) as executor:
            futures = []
            for _ in range(n_concurrent_requests):
                futures.append(executor.submit(_post, synapse))
            for future in futures:
                status_code, time_taken = future.result()
                if status_code not in reports:
                    reports[status_code] = []
                reports[status_code].append(time_taken)
    print(reports)
    return reports

if __name__ == "__main__":
    reports = benchmark_sdxl(args.n_times, args.model_name, args.max_concurrent_requests)
    plot_report(reports)
