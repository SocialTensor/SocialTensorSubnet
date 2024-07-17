import torch
import os
import openai
from logicnet.protocol import LogicSynapse
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import bittensor as bt
from concurrent import futures

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
        process_times = [response.dendrite.process_time for response in responses]
        timeout = base_synapse.timeout
        rewards = []
        for i in range(len(responses)):
            reward = (
                SIMILARITY_WEIGHT * similarities[i]
                + CORRECTNESS_WEIGHT * correctness[i]
                + PROCESSING_TIME_WEIGHT * process_times[i] / timeout
            )
            bt.logging.success(
                f"SIMILARITY: {similarities[i]}, CORRECTNESS: {correctness[i]}, PROCESSING TIME: {process_times[i]}"
            )
            rewards.append(reward)
        return rewards

    def _get_correctness(
        self, base_synapse: LogicSynapse, responses: list[LogicSynapse]
    ):
        ground_truth_answer = base_synapse.ground_truth_answer
        bt.logging.success(f"Ground truth answer: {ground_truth_answer}")
        batch_messages = [
            [
                {
                    "role": "user",
                    "content": f"{base_synapse.raw_logic_question}\n The ground truth is {ground_truth_answer}\n\n Your task is rate the correctness of this answer into 'correct' or 'incorrect'. The correct answer need have numerical or reasoning nearly equivalent to ground truth above. Just say your rating, don't reasoning anymore!.\n---{response.logic_answer}\n---",
                },
            ]
            for response in responses
        ]
        bt.logging.success(f"Batch messages: {batch_messages}")
        correctness = []
        # USE OPENAI API TO RATE THE ANSWER
        with futures.ThreadPoolExecutor() as executor:
            results = executor.map(
                lambda messages: self.openai_client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    max_tokens=32,
                    temperature=0.7,
                ),
                batch_messages,
            )
            for result in results:
                response_str = result.choices[0].message.content
                response_str = response_str.strip().lower()
                bt.logging.info(f"Correctness response: {response_str}")
                if "incorrect" in response_str:
                    correctness.append(0)
                elif "correct" in response_str:
                    correctness.append(1)
                else:
                    correctness.append(0.3)
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
