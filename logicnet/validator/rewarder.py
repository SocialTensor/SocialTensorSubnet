import torch
import openai
import sympy
import bittensor as bt
from concurrent import futures
from logicnet.protocol import LogicSynapse
from sentence_transformers import SentenceTransformer
from logicnet.utils.model_selector import model_selector

SIMILARITY_WEIGHT = 0.2
CORRECTNESS_WEIGHT = 0.8
PROCESSING_TIME_WEIGHT = -0.1

CORRECTNESS_TEMPLATE = """As an expert mathematician, evaluate how correct the response is compared to the ground truth answer. Only consider the final answer, disregarding the method or steps taken.

Instructions:
- Output only one floating-point number between 0 and 1, representing the correctness score.
- A score of 1 means completely correct, and 0 means completely incorrect.
- Do not provide any explanations or additional text.
- Consider numerical equivalence, even if the format differs (e.g., fractions vs. decimals).

Question:
---
{question}
---

Ground Truth Answer:
---
{ground_truth_answer}
---

Response:
---
{response}
---

Correctness Score (a number between 0 and 1, output only the number):"""

class LogicRewarder:
    def __init__(self, model_rotation_pool: dict):
        """
        READ HERE TO LEARN HOW VALIDATOR REWARD THE MINER
        """
        self.model_rotation_pool = model_rotation_pool
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def __call__(self, uids, responses: list[LogicSynapse], base_synapse: LogicSynapse):
        """Calculate reward for each response using similarity, correctness, and processing time.

        Args:
            uids (list[int]): List of miner UIDs.
            responses (list[LogicSynapse]): Synapse responses from miners.
            base_synapse (LogicSynapse): Base synapse containing the ground truth and raw logic question.

        Returns:
            list[float]: List of rewards for each response.
        """

        valid_uids = [
            uid for uid, response in zip(uids, responses) if response.is_success
        ]
        valid_responses = [response for response in responses if response.is_success]
        invalid_uids = [
            uid for uid, response in zip(uids, responses) if not response.is_success
        ]
        invalid_rewards = [0 for _ in invalid_uids]
        reward_logs = []
        valid_rewards = []

        if valid_uids:
            ref_ground_truth: str = self._get_ground_truth(
                base_synapse.raw_logic_question
            )
            response_texts = [response.logic_reasoning for response in valid_responses]
            similarities = self._get_similarity(ref_ground_truth, response_texts)
            correctness = self._get_correctness(base_synapse, valid_responses)
            process_times = [
                response.dendrite.process_time for response in valid_responses
            ]
            timeout = base_synapse.timeout

            for i in range(len(valid_responses)):
                reward = (
                    SIMILARITY_WEIGHT * similarities[i]
                    + CORRECTNESS_WEIGHT * correctness[i]
                    + PROCESSING_TIME_WEIGHT * min(process_times[i] / timeout, 1)
                )
                reward_logs.append(
                    {
                        "similarity": similarities[i],
                        "correctness": correctness[i],
                        "process_time": process_times[i],
                    }
                )
                # Scale up the reward
                reward = reward / 2 + 0.5
                bt.logging.debug(
                    f"[REWARDER] similarity: {similarities[i]}, correctness: {correctness[i]}, processing time: {process_times[i]}"
                )
                valid_rewards.append(reward)

        total_uids = valid_uids + invalid_uids
        rewards = valid_rewards + invalid_rewards
        return total_uids, rewards, reward_logs

    def _get_correctness(
        self, base_synapse: LogicSynapse, responses: list[LogicSynapse]
    ):
        """Calculate the correctness score for each response.

        Args:
            base_synapse (LogicSynapse): The base synapse containing the ground truth and raw logic question.
            responses (list[LogicSynapse]): List of miner responses.

        Returns:
            list[float]: List of correctness scores for each response (float between 0 and 1).
        """
        model, base_url, api_key = model_selector(self.model_rotation_pool)
        if not model:
            raise ValueError("Model ID is not valid or not provided.")
        if not base_url:
            raise ValueError("Base URL is not valid or not provided.")
        if not api_key:
            raise ValueError("API key is not valid or not provided.")
        
        openai_client = openai.OpenAI(base_url=base_url, api_key=api_key)
        bt.logging.debug(f"Initiating request with model '{model}' at base URL '{base_url}'.")

        ground_truth_answer = base_synapse.ground_truth_answer
        bt.logging.debug(f"[CORRECTNESS] Ground truth: {ground_truth_answer}")
        correctness = []
        batch_messages = []
        indices_for_llm = []

        for idx, response in enumerate(responses):
            miner_answer = response.logic_answer.strip()
            # Try programmatic comparison
            score = self._compare_numerical_answers(ground_truth_answer, miner_answer)
            if score is not None:
                correctness.append(score)
                bt.logging.debug(f"[CORRECTNESS] Used programmatic comparison for response {idx} with score {score} for answer {miner_answer} against ground truth {ground_truth_answer}")
            else:
                # Need LLM evaluation
                bt.logging.debug(f"[CORRECTNESS] Unable to use programmatic comparison. Need LLM evaluation for response {idx} with answer {miner_answer} against ground truth {ground_truth_answer}")
                correctness.append(None)  # Placeholder
                batch_messages.append([
                    {
                        "role": "user",
                        "content": CORRECTNESS_TEMPLATE.format(
                            question=base_synapse.raw_logic_question,
                            ground_truth_answer=ground_truth_answer,
                            response=miner_answer
                        ),
                    },
                ])
                # log bt.debug for what score did the LLM give
                indices_for_llm.append(idx)

        if batch_messages:
            with futures.ThreadPoolExecutor() as executor:
                for attempt in range(3):  # Retry up to 3 times
                    try:
                        results = executor.map(
                            lambda messages: openai_client.chat.completions.create(
                                model=model,
                                messages=messages,
                                max_tokens=5,
                                temperature=0,
                            ),
                            batch_messages,
                        )
                        for idx, result in zip(indices_for_llm, results):
                            response_str = result.choices[0].message.content.strip().lower()
                            bt.logging.debug(f"[CORRECTNESS] Rating: {response_str}")
                            try:
                                correctness_score = float(response_str)
                                correctness[idx] = min(max(correctness_score, 0.0), 1.0)
                            except ValueError:
                                default_score = 0.5
                                bt.logging.warning(f"Failed to parse correctness score for response {idx}. Assigning default score of {default_score}.")
                                correctness[idx] = default_score
                        break
                    
                    except openai.error.OpenAIError as e:
                        bt.logging.error(f"API request failed: {e}")
                        if attempt == 2:  # Last attempt
                            # Switch to another model, base URL, and API key
                            model, base_url, api_key = model_selector(self.model_rotation_pool)
                            if not model or not base_url or not api_key:
                                bt.logging.error("No alternative model, base URL, or API key available.")
                                for idx in indices_for_llm:
                                    correctness[idx] = 0.5
                            else:
                                openai_client = openai.OpenAI(base_url=base_url, api_key=api_key)
                                bt.logging.debug(f"Initiating request with model '{model}' at base URL '{base_url}'.")
                                try:
                                    results = executor.map(
                                        lambda messages: openai_client.chat.completions.create(
                                            model=model,
                                            messages=messages,
                                            max_tokens=5,
                                            temperature=0,
                                        ),
                                        batch_messages,
                                    )
                                    for idx, result in zip(indices_for_llm, results):
                                        response_str = result.choices[0].message.content.strip().lower()
                                        bt.logging.debug(f"[CORRECTNESS] Rating: {response_str}")
                                        try:
                                            correctness_score = float(response_str)
                                            correctness[idx] = min(max(correctness_score, 0.0), 1.0)
                                        except ValueError:
                                            default_score = 0.5
                                            bt.logging.warning(f"Failed to parse correctness score for response {idx}. Assigning default score of {default_score}.")
                                            correctness[idx] = default_score
                                    break
                                except openai.error.OpenAIError as e:
                                    bt.logging.error(f"API request failed after switching: {e}")
                                    for idx in indices_for_llm:
                                        correctness[idx] = 0.5
        return correctness

    def _compare_numerical_answers(self, ground_truth: str, miner_answer: str):
        try:
            # Remove formatting characters from the answers
            formatting_chars = ['$', '$$', '\\[', '\\]', '%']
            for char in formatting_chars:
                ground_truth = ground_truth.replace(char, '')
                miner_answer = miner_answer.replace(char, '')
            gt_value = sympy.sympify(ground_truth.strip())
            miner_value = sympy.sympify(miner_answer.strip())

            # Compute absolute difference and relative error
            abs_difference = abs(gt_value - miner_value)
            epsilon = 1e-8  # Small number to prevent division by zero
            gt_abs = abs(gt_value) + epsilon

            relative_error = abs_difference / gt_abs
            # Logs for debugging
            bt.logging.debug(f"[CORRECTNESS DEBUG FOR NUMERICAL COMPARISON] Ground truth: {gt_value}, Miner answer: {miner_value}, Absolute difference: {abs_difference}, Relative error: {relative_error}")

            # Map relative error to correctness score between 0 and 1
            # Assuming that a relative error of 0 corresponds to correctness 1
            # Larger relative errors correspond to lower correctness scores
            max_error = 1.0  # Define the maximum acceptable relative error
            correctness_score = max(0.0, 1.0 - (relative_error / max_error))

            # Clamp the correctness_score between 0 and 1
            correctness_score = min(max(correctness_score, 0.0), 1.0)

            return correctness_score
        except (sympy.SympifyError, TypeError, ZeroDivisionError) as e:
            return None

    def _get_similarity(self, ground_truth: str, responses: list[str]):
        """Calculate cosine similarity between self-generated ground truth and miner responses.

        Args:
            ground_truth (str): Ground truth generated by self.
            responses (list[str]): List of responses from miners.

        Returns:
            list[float]: List of similarity scores for each response.
        """
        ground_truth_embedding = self.embedder.encode(ground_truth)
        response_embeddings = self.embedder.encode(responses)

        # Calculate similarity
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
        """Generate self-generated ground truth based on the question.

        Args:
            question (str): Raw logic question.

        Returns:
            str: Self-generated ground truth.
        """
        messages = [
            {"role": "user", "content": question},
        ]
        model, base_url, api_key = model_selector(self.model_rotation_pool)
        if not model:
            raise ValueError("Model ID is not valid or not provided.")
        if not base_url:
            raise ValueError("Base URL is not valid or not provided.")
        if not api_key:
            raise ValueError("API key is not valid or not provided.")

        openai_client = openai.OpenAI(base_url=base_url, api_key=api_key)
        bt.logging.debug(f"Initiating request with model '{model}' at base URL '{base_url}'.")

        response = ""
        for attempt in range(3):  # Retry up to 3 times
            try:
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.7,
                )
                response = response.choices[0].message.content
                bt.logging.debug(f"[SIMILARITY] Self-generated ground truth: {response}")
                return response  # Return response if successful
            
            except openai.error.OpenAIError as e:
                bt.logging.error(f"API request failed on attempt {attempt + 1}: {e}")
                if attempt == 2:  # Last attempt
                    # Switch to another model, base URL, and API key
                    model, base_url, api_key = model_selector(self.model_rotation_pool)
                    if not model or not base_url or not api_key:
                        bt.logging.error("No alternative model, base URL, or API key available.")

                    else:
                        openai_client = openai.OpenAI(base_url=base_url, api_key=api_key)
                        bt.logging.debug(f"Initiating request with model '{model}' at base URL '{base_url}'.")
                        try:
                            response = openai_client.chat.completions.create(
                                model=model,
                                messages=messages,
                                max_tokens=1024,
                                temperature=0.7,
                            )
                            response = response.choices[0].message.content
                            bt.logging.debug(f"[SIMILARITY] Self-generated ground truth: {response}")
                            return response
                        except openai.error.OpenAIError as e:
                            bt.logging.error(f"API request failed after switching: {e}")

        return response
