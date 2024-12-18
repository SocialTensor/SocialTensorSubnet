import bittensor as bt
import pydantic
from typing import Union


class Information(bt.Synapse):
    """Information synapse for miner to send information to the validator. It won't be blacklisted by miner"""

    request_dict: dict = {}
    response_dict: dict = {}


class LogicSynapse(bt.Synapse):
    """
    Logic Synapse for the LogicNet protocol
    """

    # MINER NEED TO FILL THIS INFORMATION
    logic_question: str = pydantic.Field(
        "",
        description="Logic question to be answered by miner. It can be noised question from the raw logic question from synthetic loop."
    )
    logic_answer: Union[str, object] = pydantic.Field(
        "", description="Short logic answer as a summary of the logic reasoning."
    )
    logic_reasoning: str = pydantic.Field(
        "",
        description="Reasoning when answering the logic question"
    )

    # ONLY VISIBLE TO VALIDATOR
    raw_logic_question: str = pydantic.Field(
        "",
        description="If this synapse from synthetic loop, this field will contain the raw logic question from the dataset."
    )
    ground_truth_answer: Union[str, object] = pydantic.Field(
        "",
        description="Ground truth answer for the logic question. Very short, only the answer."
    )

    # SYNAPSE INFORMATION
    category: str = pydantic.Field(
        "",
        description="One of the categories in the Validator main."
    )
    timeout: int = pydantic.Field(
        64,
        description="Timeout for the miner to answer the logic question."
    )

    # UNIQUE TASK UID FOR CORRELATION
    task_uid: str = pydantic.Field(
        "",
        description="A unique identifier for this specific logic task, assigned by the validator."
    )

    def miner_synapse(self):
        """
        HIDE THE GROUND TRUTH AND RAW LOGIC QUESTION FROM THE MINER
        """
        self.raw_logic_question = ""
        self.ground_truth_answer = None
        return self

    def deserialize_response(self):
        return {
            "logic_answer": self.logic_answer,
            "logic_reasoning": self.logic_reasoning
        }


class LogicRequest(pydantic.BaseModel):
    """
    Logic Synapse for the LogicNet protocol
    """

    # MINER NEED TO FILL THIS INFORMATION
    logic_question: str = pydantic.Field(
        "",
        description="Logic question to be answered by miner. It can be noised question from the raw logic question from synthetic loop.",
    )
    logic_answer: Union[str, object] = pydantic.Field(
        "", description="Short logic answer as a summary of the logic reasoning."
    )
    logic_reasoning: str = pydantic.Field(
        "",
        description="Reasoning when answering the logic question",
    )

    # SYNAPSE INFORMATION
    category: str = pydantic.Field(
        "",
        description="One of the categories in the Validator main.",
    )
    timeout: int = pydantic.Field(
        64,
        description="Timeout for the miner to answer the logic question.",
    )
