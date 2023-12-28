import yaml
import os

CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

MODEL_CONFIG = yaml.load(open("model_config.yaml"), yaml.FullLoader)

if __name__ == "__main__":
    for model_name, config in MODEL_CONFIG.items():
        print("Downloading", model_name)
        file = os.path.join(CKPT_DIR, model_name)
        url = config['checkpoint_url']
        # download file from url to file
        command = f"curl -L \"{url}\" --output \"{file}.safetensors\""
        print(command)
        os.system(command)
