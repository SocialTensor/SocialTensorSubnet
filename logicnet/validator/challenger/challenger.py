# Challenge for Synthetic Request
import openai
import random
import os
from logicnet.protocol import LogicSynapse
from dotenv import load_dotenv
import bittensor as bt
from .human_noise import get_condition
from .math_generator.topics import TOPICS as topics
import mathgenerator

load_dotenv(override=True)
MODEL = os.getenv("CHALLENGE_MODEL", "gpt-3.5-turbo")
BASE_URL = os.getenv("CHALLENGE_BASE_URL", "https://api.openai.com/v1")
KEY = os.getenv("CHALLENGE_KEY")

print(MODEL, BASE_URL)


class LogicChallenger:
    def __init__(self):
        self.openai_client = openai.OpenAI(base_url=BASE_URL, api_key=KEY)

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
        bt.logging.info(f"Generated atom math problem: {atom_problem}")
        subtopic = subtopic.replace("_", " ").capitalize()
        topic = topic.replace("_", " ").capitalize()
        atom_problem = atom_problem.replace("$", "").strip()
        atom_problem = f"Find the solution of this math problem from topic {subtopic}, {topic}:\n---\n{atom_problem}\n---\n"
        synapse.raw_logic_question = atom_problem

        synapse.ground_truth_answer = str(atom_answer).replace("$", "").strip()

        bt.logging.debug(f"Generated atom math answer: {atom_answer}")

        return atom_problem

    def get_revised_math_question(self, math_problem: str, conditions: dict) -> str:
        prompt = "Please paraphrase by adding word or expression to this question as if you were a {profile} who is {mood} and write in a {tone} tone. You can use incorrect grammar, typo or add more context! Don't add your solution! Just say the revised version, you don't need to be polite.".format(
            **conditions
        )
        bt.logging.debug(f"Revising prompt: {prompt}")
        messages = [
            {
                "role": "user",
                "content": "Generate a math problem that required logic to solve.",
            },
            {"role": "assistant", "content": math_problem},
            {
                "role": "user",
                "content": prompt,
            },
        ]
        response = self.openai_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=256,
            temperature=0.5,
        )
        response = response.choices[0].message.content
        bt.logging.info(f"Generated revised math question: {response}")
        return response
