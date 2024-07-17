import bittensor as bt
import pydantic
from typing import Union


class Information(bt.Synapse):
    request_dict: dict = {}
    response_dict: dict = {}


class LogicSynapse(bt.Synapse):
    """
    Attributes:
    TODO
    """

    logic_question: str = pydantic.Field("", title="Logic Question", description="")
    logic_answer: Union[str, object] = pydantic.Field(
        "", title="Logic Reasoning", description=""
    )
    logic_reasoning: str = pydantic.Field(
        "",
        title="Logic Answer",
        description="",
    )

    logic_answer_type: str = pydantic.Field(
        "",
        title="Logic Answer Type",
        description="",
    )

    logic_answer_preprocess_function: str = pydantic.Field(
        "",
        title="Logic Answer Preprocess Function",
        description="",
    )

    category: str = pydantic.Field(
        "",
        title="Category",
        description="",
    )

    raw_logic_question: str = pydantic.Field(
        "",
        title="Raw Logic Question",
        description="",
    )

    timeout: int = pydantic.Field(
        12,
        title="Timeout",
        description="",
    )

    logic_ground_truth: Union[str, object] = pydantic.Field(
        "",
        title="Logic Ground Truth",
        description="",
    )

    def miner_synapse(self):
        self.raw_logic_question = ""
        self.logic_ground_truth = ""
        return self
