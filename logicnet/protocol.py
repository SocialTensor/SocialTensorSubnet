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

    # MINER KNOWLEDGE
    logic_question: str = pydantic.Field("", title="Logic Question", description="")
    logic_answer: Union[str, object] = pydantic.Field(
        "", title="Logic Reasoning", description=""
    )
    logic_reasoning: str = pydantic.Field(
        "",
        title="Logic Answer",
        description="",
    )

    # VALIDATOR KNOWLEDGE
    raw_logic_question: str = pydantic.Field(
        "",
        title="Raw Logic Question",
        description="",
    )
    ground_truth_answer: Union[str, object] = pydantic.Field(
        "",
        title="Logic Ground Truth",
        description="",
    )

    # SYNAPSE INFORMATION
    category: str = pydantic.Field(
        "",
        title="Category",
        description="",
    )
    timeout: int = pydantic.Field(
        12,
        title="Timeout",
        description="",
    )

    def miner_synapse(self):
        self.raw_logic_question = ""
        self.ground_truth_answer = None
        return self
