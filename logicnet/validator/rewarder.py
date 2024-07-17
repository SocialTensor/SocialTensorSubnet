import torch
import os
import openai
from logicnet.protocol import LogicSynapse
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import bittensor as bt
from latex2sympy2 import latex2sympy

load_dotenv()
# RECOMMENDED MODEL GPT4 - OPENAI
MODEL = os.getenv("REWARD_MODEL", "gpt-3.5-turbo")
BASE_URL = os.getenv("REWARD_BASE_URL", "https://api.openai.com/v1")
KEY = os.getenv("REWARD_KEY")

SIMILARITY_WEIGHT = 0.2
CORRECTNESS_WEIGHT = 0.8
PROCESSING_TIME_WEIGHT = -0.1


class LogicRewarder:
    def __init__(self):
        self.openai_client = openai.OpenAI(base_url=BASE_URL, api_key=KEY)
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def __call__(self, uids, responses: list[LogicSynapse], base_synapse: LogicSynapse):
        ref_ground_truth: str = self._get_ground_truth(base_synapse.raw_logic_question)
        response_texts = [response.logic_reasoning for response in responses]
        similarities = self._get_similarity(ref_ground_truth, response_texts)
        correctness = self._get_correctness(base_synapse, responses)
        process_times = [response.process_time for response in responses]

        rewards = []
        for i in range(len(responses)):
            reward = (
                SIMILARITY_WEIGHT * similarities[i]
                + CORRECTNESS_WEIGHT * correctness[i]
                + PROCESSING_TIME_WEIGHT * process_times[i]
            )
            rewards.append(reward)
        return rewards

    def _get_correctness(
        self, base_synapse: LogicSynapse, responses: list[LogicSynapse]
    ):
        logic_ground_truth = base_synapse.logic_ground_truth
        logic_answer_type = base_synapse.logic_answer_type

        correctness = []
        for response in responses:
            try:
                if logic_answer_type == "sympy":
                    logic_answer = latex2sympy(response.logic_answer)
                    is_correct = logic_ground_truth.compare(logic_answer)
                elif logic_answer_type == "python_object":
                    logic_answer = eval(response.logic_answer)
                    is_correct = logic_ground_truth == logic_answer
                elif logic_answer_type == "str":
                    is_correct = self._get_is_correct_str_answer(
                        logic_ground_truth,
                        response.logic_answer,
                        base_synapse.raw_logic_question,
                    )
                else:
                    is_correct = 0
            except Exception as e:
                bt.logging.debug(f"Error comparing answer: {e}")
                is_correct = 0
            correctness.append(is_correct)
        return correctness

    def _get_is_correct_str_answer(
        self, ground_truth: str, response: str, question: str
    ):
        messages = [
            {
                "role": "user",
                "content": question,
            },
            {
                "role": "assistant",
                "content": ground_truth,
            },
            {
                "role": "user",
                "content": f"Is this solution equivalent? Say only Yes or No.\n---\n{response}\n---",
            },
        ]
        response = self.openai_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=32,
            temperature=0.7,
        )
        response_str = response.choices[0].message.content
        response_str = response_str.strip().lower()
        bt.logging.debug(f"Correctness response: {response_str}")
        if "yes" in response_str:
            return 1
        else:
            return 0

    def _get_similarity(self, ground_truth: str, responses: list[str]):
        ground_truth_embedding = self.embedder.encode(ground_truth)
        response_embeddings = self.embedder.encode(responses)

        # calculate similarity
        similarities = []
        for response_embedding in response_embeddings:
            similarity = torch.nn.functional.cosine_similarity(
                torch.tensor(ground_truth_embedding),
                torch.tensor(response_embedding),
                dim=0,
            )
            similarities.append(similarity.item())
        return similarities

    def _get_ground_truth(self, question: str):
        messages = [
            {"role": "user", "content": question},
        ]
        response = self.openai_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
        )
        response = response.choices[0].message.content
        bt.logging.info(f"Generated ground truth: {response}")
        return response
