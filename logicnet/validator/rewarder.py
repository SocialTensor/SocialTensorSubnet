import torch
import openai
import sympy
import bittensor as bt
from concurrent import futures
from logicnet.protocol import LogicSynapse
from sentence_transformers import SentenceTransformer
from logicnet.utils.model_selector import model_selector
from logicnet.utils.regex_helper import extract_numerical_part
from logicnet.validator.prompt import DETECT_TRICK_TEMPLATE, CORRECTNESS_TEMPLATE

SIMILARITY_WEIGHT = 0.2
CORRECTNESS_WEIGHT = 0.8
PROCESSING_TIME_WEIGHT = -0.1



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
            task_uid (int): Unique task UID.
            uids (list[int]): List of miner UIDs.
            responses (list[LogicSynapse]): Synapse responses from miners.
            base_synapse (LogicSynapse): Base synapse containing the ground truth and raw logic question.

        Returns:
            list[float]: List of rewards for each response.
        """
        # Get the unique task UID from the base_synapse
        task_uid = base_synapse.task_uid

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
                reward_info = {
                    "task_uid": task_uid,  # Include the task_uid in the reward log
                    "similarity": similarities[i],
                    "correctness": correctness[i],
                    "process_time": process_times[i],
                }
                reward_logs.append(reward_info)

                # Scale up the reward
                reward = reward / 2 + 0.5
                bt.logging.debug(
                    f"[REWARDER][{task_uid}] similarity: {similarities[i]}, correctness: {correctness[i]}, process_time: {process_times[i]}, final_reward: {reward}"
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
        batch_llm_inputs = []
        indices_for_llm = []

        for idx, response in enumerate(responses):
            miner_answer = response.logic_answer.strip()
            # Try programmatic comparison
            score = self._compare_numerical_answers(ground_truth_answer, miner_answer)
            if score is not None:
                correctness.append(score)
                bt.logging.debug(f"[CORRECTNESS] Used programmatic comparison for response {idx} with score {score}")
            else:
                # Need LLM evaluation
                bt.logging.debug(f"[CORRECTNESS] Unable to use programmatic comparison. Need LLM evaluation for response {idx}")
                correctness.append(0)  # Placeholder
                batch_llm_inputs.append({
                    "question": base_synapse.raw_logic_question,
                    "ground_truth_answer": ground_truth_answer,
                    "response": miner_answer
                })
                # log bt.debug for what score did the LLM give
                indices_for_llm.append(idx)

        if batch_llm_inputs:
            with futures.ThreadPoolExecutor() as executor:
                for attempt in range(3):  # Retry up to 3 times
                    try:
                        llm_scores = executor.map(
                            lambda inputs: self._get_correctness_by_llm(
                                question=inputs["question"],
                                ground_truth=inputs["ground_truth_answer"],
                                response=inputs["response"],
                                model_name=model,
                                openai_client=openai_client,
                            ),
                            batch_llm_inputs,
                        )
                        for idx, score in zip(indices_for_llm, llm_scores):
                            bt.logging.debug(f"[CORRECTNESS] Rating: {score}")
                            correctness[idx] = score
                        break
                    except Exception as e:
                        bt.logging.error(f"Error in compute score by llm model: {e}")
                        for idx in indices_for_llm:
                            correctness[idx] = 0.5
        return correctness
    

    def _get_correctness_by_llm(self, question: str, ground_truth: str, response: str, model_name: str, openai_client: openai.OpenAI):
        """Calculate the correctness score for a single response using LLM.

        Args:
            question (str): Raw logic question.
            ground_truth (str): Ground truth answer.
            response (str): Miner's answer.
            model_name (str): Model name for the LLM.
            openai_client (openai.OpenAI): OpenAI client for API requests.

        Returns:
            float: Correctness score for the response (float between 0 and 1).
        """

        ## check trick case
        try:
            response_str = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": DETECT_TRICK_TEMPLATE.format(
                            response=response
                        ),
                    },
                ],
                max_tokens=5,
                temperature=0,
            ).choices[0].message.content.strip().lower()
            bt.logging.debug(f"[CORRECTNESS] Trick detection: {response_str}")
            if "yes" in response_str:
                return 0
        except Exception as e:
            bt.logging.error(f"API request failed: {e}")
        
        try:
            response_str = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": CORRECTNESS_TEMPLATE.format(
                            question=question,
                            ground_truth_answer=ground_truth,
                            response=response
                        ),
                    },
                ],
                max_tokens=5,
                temperature=0,
            ).choices[0].message.content.strip().lower()
            bt.logging.debug(f"[CORRECTNESS] Rating: {response_str}")
            try:
                correctness_score = float(response_str)
                return min(max(correctness_score, 0.0), 1.0)
            except ValueError:
                bt.logging.warning(f"Failed to parse correctness score. Assigning default score of 0.5.")
                return 0.5
        except openai.OpenAIError as e:
            bt.logging.error(f"API request failed: {e}")
            # Switch to another model, base URL, and API key
            model, base_url, api_key = model_selector(self.model_rotation_pool)
            if not model or not base_url or not api_key:
                bt.logging.error("No alternative model, base URL, or API key available.")
                return 0.5
            else:
                openai_client = openai.OpenAI(base_url=base_url, api_key=api_key)
                bt.logging.debug(f"Initiating request with model '{model}' at base URL '{base_url}'.")
                response_str = openai_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": CORRECTNESS_TEMPLATE.format(
                                question=question,
                                ground_truth_answer=ground_truth,
                                response=response
                            ),
                        },
                    ],
                    max_tokens=5,
                    temperature=0,
                ).choices[0].message.content.strip().lower()
                bt.logging.debug(f"[CORRECTNESS] Rating: {response_str}")
                try:
                    correctness_score = float(response_str)
                    return min(max(correctness_score, 0.0), 1.0)
                except ValueError:
                    bt.logging.warning(f"Failed to parse correctness score. Assigning default score of 0.5.")
                    return 0.5

    def _compare_numerical_answers(self, ground_truth: str, miner_answer: str):
        try:
            # Remove formatting characters from the answers
            formatting_chars = ['$', '$$', '\\[', '\\]', '%']
            for char in formatting_chars:
                ground_truth = ground_truth.replace(char, '')
                miner_answer = miner_answer.replace(char, '')

            # Extract numerical values
            gt_value_str = extract_numerical_part(ground_truth)
            miner_value_str = extract_numerical_part(miner_answer)

            if gt_value_str is None or miner_value_str is None:
                raise ValueError("No numerical value found in one of the answers.")

            gt_value = sympy.sympify(gt_value_str)
            miner_value = sympy.sympify(miner_value_str)

            abs_difference = abs(gt_value - miner_value)
            epsilon = 1e-8
            gt_abs = abs(gt_value) + epsilon
            relative_error = abs_difference / gt_abs
            # Logs for debugging
            bt.logging.debug(f"[CORRECTNESS DEBUG FOR NUMERICAL COMPARISON]: Absolute difference: {abs_difference}, Relative error: {relative_error}")

            correctness_score = max(0.0, 1.0 - relative_error)
            correctness_score = min(correctness_score, 1.0)
            return correctness_score
        except Exception as e:
            # Log the problematic input for debugging
            bt.logging.warning(
                f"Failed to sympify numerical answers.\nError: {e}"
            )
            # Return None so that LLM-based correctness check will be used.
            return None

    def _get_similarity(self, ground_truth: str, responses: list[str]):
        """Calculate cosine similarity between self-generated ground truth and miner responses.

        Args:
            ground_truth (str): Ground truth generated by self.
            responses (list[str]): List of responses from miners.

        Returns:
            list[float]: List of similarity scores for each response.
        """
        try:
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
        except Exception as e:
            bt.logging.warning(f"Failed to calculate similarity.\nError: {e}")
            return [0.5] * len(responses)

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
            
            except openai.OpenAIError as e:
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
                        except openai.OpenAIError as e:
                            bt.logging.error(f"API request failed after switching: {e}")

        return response