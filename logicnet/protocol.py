import bittensor as bt
import pydantic


class Information(bt.Synapse):
    request_dict: dict = {}
    response_dict: dict = {}


class LogicSynapse(bt.Synapse):
    """
    Attributes:
    TODO
    """

    logic_question: str = pydantic.Field("", title="Logic Question", description="")

    logic_answer: str = pydantic.Field(
        "",
        title="Logic Answer",
        description="",
    )
