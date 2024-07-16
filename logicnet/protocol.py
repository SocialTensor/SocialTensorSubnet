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

    def miner_synapse(self):
        self.raw_logic_question = ""
