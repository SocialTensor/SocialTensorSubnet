# Challenge for Synthetic Request
import openai
import random
import mathgenerator
import bittensor as bt
from logicnet.protocol import LogicSynapse
from .human_noise import get_condition
from .math_generator.topics import TOPICS as topics
from logicnet.utils.model_selector import model_selector


class LogicChallenger:
    def __init__(self, model_rotation_pool: dict):
        self.model_rotation_pool = model_rotation_pool
        

    def __call__(self, synapse: LogicSynapse) -> LogicSynapse:
        self.get_challenge(synapse)
        return synapse

    def get_challenge(self, synapse: LogicSynapse):
        logic_problem = self.get_atom_math_problem(synapse)
        conditions: dict = get_condition()
        revised_logic_question: str = self.get_revised_math_question(
            logic_problem, conditions
        )
        synapse.logic_question = revised_logic_question

    def get_atom_math_problem(self, synapse: LogicSynapse) -> str:
        selected_topic = random.choice(topics)
        subtopic = selected_topic["subtopic"]
        topic = selected_topic["topic"]
        bt.logging.debug(f"Using {mathgenerator.__name__} to generate math problem")
        atom_problem, atom_answer = eval(f"mathgenerator.{topic}.{subtopic}()")
        subtopic = subtopic.replace("_", " ").capitalize()
        topic = topic.replace("_", " ").capitalize()
        atom_problem = atom_problem.replace("$", "").strip()
        atom_problem = f"Find the solution of this math problem:\n---\nTopic: {topic}, Subtopic: {subtopic}.\n{atom_problem}\n---\n"
        bt.logging.debug(f"Generated atom math problem: {atom_problem}")
        synapse.raw_logic_question = atom_problem

        synapse.ground_truth_answer = str(atom_answer).replace("$", "").strip()

        bt.logging.debug(f"Generated atom math answer: {atom_answer}")

        return atom_problem

    def get_revised_math_question(self, math_problem: str, conditions: dict) -> str:
        # prompt = "Please paraphrase by adding word or expression to this question as if you were a {profile} who is {mood} and write in a {tone} tone. You can use incorrect grammar, typo or add more context! Don't add your solution! Just say the revised version, you don't need to be polite.".format(
        #     **conditions
        # )
        
        prompt = (
            "As a {profile} who is feeling {mood}, please rephrase the following math problem "
            "in a {tone} tone. Write it as you would naturally ask the question. "
            "Do not include the solution or add unnecessary context."
        ).format(**conditions)
        
        bt.logging.debug(f"Revising prompt: {prompt}")
        
        # messages = [
        #     {
        #         "role": "user",
        #         "content": "Generate a math problem that required logic to solve.",
        #     },
        #     {"role": "assistant", "content": math_problem},
        #     {
        #         "role": "user",
        #         "content": prompt,
        #     },
        # ]
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are simulating various human personas asking math problems. "
                    "Rephrase the following math problem as the specified persona, "
                    "ensuring the question sounds natural and appropriate for that individual."
                ),
            },
            {"role": "assistant", "content": math_problem},
            {"role": "user", "content": prompt},
        ]

        max_attempts = 3

        for attempt in range(max_attempts):
            model, base_url, api_key = model_selector(self.model_rotation_pool)
            if not model or not base_url or not api_key:
                raise ValueError("Model configuration is incomplete.")

            openai_client = openai.OpenAI(base_url=base_url, api_key=api_key)
            bt.logging.debug(f"Initiating request with model '{model}' at base URL '{base_url}'.")

            try:
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=256,
                    temperature=0.7,
                )
                revised_question = response.choices[0].message.content.strip()
                bt.logging.debug(f"Generated revised math question: {revised_question}")
                return revised_question
            
            except openai.error.OpenAIError as e:
                bt.logging.error(f"OpenAI API request failed (attempt {attempt + 1}): {e}")
                if attempt == max_attempts - 1:
                    raise RuntimeError("Failed to get a response after multiple attempts with different models.")
                bt.logging.info("Switching to a different model configuration.")