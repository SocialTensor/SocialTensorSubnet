from diffusers import AutoPipelineForText2Image, EulerAncestralDiscreteScheduler

def load_diffuser_t2i(model_name):
    model_config = MODEL_CONFIG[model_name]
    checkpoint_file = model_config["save_dir"]
    model_type = model_config["model_type"]
    if model_type == "sd-1.5":
        pipe = AutoPipelineForText2Image.from_pretrained(model_name, use_safetensors=True)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    return pipe