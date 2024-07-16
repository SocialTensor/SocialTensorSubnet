import torch
import os
import openai
from logicnet.protocol import LogicSynapse
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import bittensor as bt

load_dotenv()
# RECOMMENDED MODEL GPT4 - OPENAI
MODEL = os.getenv("REWARD_MODEL", "gpt-3.5-turbo")
BASE_URL = os.getenv("REWARD_BASE_URL", "https://api.openai.com/v1")
KEY = os.getenv("REWARD_KEY")

SIMILARITY_WEIGHT = 0.5
CORRECTNESS_WEIGHT = 0.4
PROCESSING_TIME_WEIGHT = -0.1


class LogicRewarder:
    def __init__(self):
        self.openai_client = openai.OpenAI(base_url=BASE_URL, api_key=KEY)
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def __call__(self, uids, responses: list[LogicSynapse], base_synapse: LogicSynapse):
        ground_truth = self._get_ground_truth(base_synapse.raw_logic_question)
        response_texts = [response.logic_answer for response in responses]
        similarities = self._get_similarity(ground_truth, response_texts)
        correctness_scores = self._get_correct_score(
            base_synapse.raw_logic_question, ground_truth, response_texts
        )
        processing_times = [response.dendrite.process_time for response in responses]
        timeout = base_synapse.timeout
        final_scores = []
        for similarity, correctness, process_time in zip(
            similarities, correctness_scores, processing_times
        ):
            bt.logging.info(
                f"Similarity: {similarity:.2f}, Correctness: {correctness:.2f}, Process time: {process_time:.2f}"
            )
            final_score = (
                SIMILARITY_WEIGHT * similarity
                + CORRECTNESS_WEIGHT * correctness
                + PROCESSING_TIME_WEIGHT * process_time / timeout
            )
            final_scores.append(final_score)

        return final_scores

    def _get_correct_score(self, question, ground_truth, responses_texts):
        scores = []
        for response in responses_texts:
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
                    "content": f"Is this solution correct? Generate only Yes or No.\n---\n{response}\n---",
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
                scores.append(1)
            else:
                scores.append(0)

        return scores

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
