from PIL import Image
from .base_model import BaseModel

class OpenCategoryPipeline(BaseModel):
    def load_model(self, supporting_pipelines, **kwargs):
        ### Miner update code here
        pipelines = {
            "txt2img": None
        }

        def inference_function(*args, **kwargs) -> Image.Image:
            pipeline_type = kwargs["pipeline_type"]
            pipeline = pipelines.get("pipeline_type")
            if not pipeline:
                raise ValueError(f"Pipeline type {pipeline_type} is not supported")
            
            output = pipeline(*args, **kwargs)
            output = None
            if output is None:
                return Image.new("RGB", (512, 512), (255, 255, 255))
            return output.images[0]

        return inference_function
