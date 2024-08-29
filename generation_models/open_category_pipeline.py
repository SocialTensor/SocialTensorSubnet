from PIL import Image
from .base_model import BaseModel
import diffusers
class OpenCategoryPipeline(BaseModel):
    def load_model(self, repo_id, supporting_pipelines, **kwargs):
        ### Miner update code here. Currently, only the txt2img pipeline is applied, using the default diffusers.DiffusionPipeline. 
        ### If using the default pipeline, update repo_id in generation_models/configs/model_config.yaml to change model.

        txt2img_pipeline = diffusers.DiffusionPipeline.from_pretrained(repo_id)
        txt2img_pipeline.to("cuda")
        pipelines = {
            "txt2img": txt2img_pipeline
        }

        def inference_function(*args, **kwargs) -> Image.Image:
            pipeline_type = kwargs["pipeline_type"]
            pipeline = pipelines.get(pipeline_type)
            if not pipeline:
                raise ValueError(f"Pipeline type {pipeline_type} is not supported")
            
            output = pipeline(*args, **kwargs)
            output = None
            if output is None:
                return Image.new("RGB", (512, 512), (255, 255, 255))
            return output.images[0]

        return inference_function
