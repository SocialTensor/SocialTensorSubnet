import typing
import bittensor as bt
from bittensor.synapse import Synapse
import pydantic
from PIL import Image
import io
import base64


def pil_image_to_base64(image):
    image_stream = io.BytesIO()
    image.save(image_stream, format="PNG")
    base64_image = base64.b64encode(image_stream.getvalue()).decode("utf-8")

    return base64_image


class ImageGenerating(bt.Synapse):
    prompt: str = pydantic.Field(
        default="",
        title="Prompt",
        description="Requested prompt for text to image generating",
    )
    seed: int = pydantic.Field(
        default=0, title="Seed", description="Seed for deterministic generation"
    )
    images: typing.List[str] = pydantic.Field(
        default=[], title="Images", description="Output of text to image model"
    )
    pipeline_params: dict = pydantic.Field(
        default={},
        title="Pipeline Parameters",
        description="Additional generating params",
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

    def deserialize(self) -> typing.List[str]:
        return self.images
