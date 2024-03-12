import bittensor as bt
import pydantic


class ImageGenerating(bt.Synapse):
    prompt: str = pydantic.Field(
        default="",
        title="Prompt",
        description="Prompt for generation",
    )
    seed: int = pydantic.Field(
        default=0,
        title="Seed",
        description="Seed for generation",
    )
    model_name: str = pydantic.Field(
        default="",
        title="",
        description="Name of the model used for generation",
    )
    conditional_image: str = pydantic.Field(
        default="",
        title="Base64 Image",
        description="Base64 encoded image",
    )
    pipeline_type: str = pydantic.Field(
        default="txt2img",
        title="Pipeline Type",
        description="Type of pipeline used for generation, eg: txt2img, img2img, controlnet_txt2img",
    )
    pipeline_params: dict = pydantic.Field(
        default={},
        title="Pipeline Params",
        description="Dictionary of additional parameters for diffusers pipeline",
    )
    request_dict: dict = pydantic.Field(
        default={},
        title="Dictionary contains request",
        description="Dict contains arbitary information",
    )
    response_dict: dict = pydantic.Field(
        default={},
        title="Dictionary contains response",
        description="Dict contains arbitary information",
    )
    image = pydantic.Field(
        default="",
        title="Base64 Image",
        description="Base64 encoded image",
    )

    def limit_params(self):
        for k, v in self.pipeline_params.items():
            if k == "num_inference_steps":
                self.pipeline_params[k] = min(50, v)
        self.pipeline_params = self.pipeline_params

    def deserialize(self) -> dict:
        return {
            "prompt": self.prompt,
            "seed": self.seed,
            "model_name": self.model_name,
            "pipeline_type": self.pipeline_type,
            "pipeline_params": self.pipeline_params,
            "conditional_image": self.conditional_image,
            "image": self.image,
            "response_dict": self.response_dict,
        }
