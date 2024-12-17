import random

def model_selector(model_rotation_pool):
    # Filter out entries with "no use" or where the model is "null"
    valid_models = {k: v for k, v in model_rotation_pool.items() if v != "no use" and v[2] != "null"}
    
    # Select a random model from the valid ones
    model_key = random.choice(list(valid_models.keys()))
    base_url, api_key, model = valid_models[model_key]

    # Return the selected model details
    return model, base_url, api_key