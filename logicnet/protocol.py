import bittensor as bt
import pydantic
from generation_models.utils import base64_to_pil_image
import typing
import requests


class Information(bt.Synapse):
    request_dict: dict = {}
    response_dict: dict = {}


class LogicSynapse(bt.Synapse):
    """
    Attributes:
    TODO
    """

    logic_question: str = pydantic.Field(
        "",
        title="Logic Question",
        description=""
    )

    logic_answer: str = pydantic.Field(
        "",
        title="Logic Answer",
        description="",
    )
