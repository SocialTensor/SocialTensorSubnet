from fastapi import FastAPI
import yaml

app = FastAPI()

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

@app.get("/get-config")
def get_config():
    config = read_yaml("admin/model_config.yaml")
    return config
