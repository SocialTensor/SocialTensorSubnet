import typing
import bittensor as bt
from bittensor.synapse import Synapse
import pydantic


class NicheImageProtocol(bt.Synapse):
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
    pipeline_params: dict = pydantic.Field(
        default={},
        title="Pipeline Params",
        description="Dictionary of additional parameters for diffusers pipeline",
    )

    def limit_params(self):
        for k, v in self.pipeline_params.items():
            if k == "num_inference_steps":
                self.pipeline_params[k] = min(50, v)
        self.pipeline_params = self.pipeline_params

    def deserialize(self) -> dict:
        return self.response_dict


class TextToImage(NicheImageProtocol):
    prompt: str = pydantic.Field(
        default="",
        title="Prompt",
        description="Requested prompt for text to image generating",
    )
    seed: int = pydantic.Field(
        default=0, title="Seed", description="Seed for deterministic generation"
    )
    model_name: str = pydantic.Field(
        default="",
        title="Model Name",
        description="Name of the model used for generation",
    )
    category_name: str = pydantic.Field(
        default="",
        title="Category Name",
        description="Name of the category used for generation",
    )
    image: str = pydantic.Field(
        default="",
        title="Image",
        description="Output of text to image model in base64 format",
    )
    pipeline_params: dict = pydantic.Field(
        default={},
        title="Pipeline Params",
        description="Dictionary of additional parameters for diffusers pipeline",
    )

    def deserialize(self) -> dict:
        return {
            "prompt": self.prompt,
            "seed": self.seed,
            "model_name": self.model_name,
            "category_name": self.category_name,
            "image": self.image,
            "pipeline_params": self.pipeline_params,
        }


class ImageToImage(NicheImageProtocol):
    prompt: str = pydantic.Field(
        default="",
        title="Prompt",
        description="Requested prompt for text to image generating",
    )
    seed: int = pydantic.Field(
        default=0, title="Seed", description="Seed for deterministic generation"
    )
    model_name: str = pydantic.Field(
        default="",
        title="Model Name",
        description="Name of the model used for generation",
    )
    category_name: str = pydantic.Field(
        default="",
        title="Category Name",
        description="Name of the category used for generation",
    )
    conditional_image: str = pydantic.Field(
        default="",
        title="Conditional Image",
        description="Initial image in base64 format",
    )
    pipeline_params: dict = pydantic.Field(
        default={},
        title="Pipeline Params",
        description="Dictionary of additional parameters for diffusers pipeline",
    )
    image: str = pydantic.Field(
        default="",
        title="Image",
        description="Output of text to image model in base64 format",
    )

    def deserialize(self) -> dict:
        return {
            "prompt": self.prompt,
            "seed": self.seed,
            "model_name": self.model_name,
            "category_name": self.category_name,
            "conditional_image": self.conditional_image,
            "pipeline_params": self.pipeline_params,
            "image": self.image,
        }


class ControlNetTextToImage(NicheImageProtocol):
    prompt: str = pydantic.Field(
        default="",
        title="Prompt",
        description="Requested prompt for text to image generating",
    )
    seed: int = pydantic.Field(
        default=0, title="Seed", description="Seed for deterministic generation"
    )
    model_name: str = pydantic.Field(
        default="",
        title="Model Name",
        description="Name of the model used for generation",
    )
    category_name: str = pydantic.Field(
        default="",
        title="Category Name",
        description="Name of the category used for generation",
    )
    conditional_image: str = pydantic.Field(
        default="",
        title="Controlnet Image",
        description="Controlnet image in base64 format",
    )
    pipeline_params: dict = pydantic.Field(
        default={},
        title="Pipeline Params",
        description="Dictionary of additional parameters for diffusers pipeline",
    )

    def deserialize(self) -> dict:
        return {
            "prompt": self.prompt,
            "seed": self.seed,
            "model_name": self.model_name,
            "category_name": self.category_name,
            "conditional_image": self.controlnet_image,
            "pipeline_params": self.pipeline_params,
            "image": self.image,
        }
