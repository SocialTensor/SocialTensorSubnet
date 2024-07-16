# Challenge for Synthetic Request
import openai
import random
import os
from logicnet.protocol import LogicSynapse
from dotenv import load_dotenv

load_dotenv()
MODEL = os.getenv("CHALLENGE_MODEL", "gpt-3.5-turbo")
BASE_URL = os.getenv("CHALLENGE_BASE_URL", "https://api.openai.com/v1")
KEY = os.getenv("CHALLENGE_KEY")


class LogicChallenger:
    def __init__(self):
        self.openai_client = openai.OpenAI(base_url=BASE_URL, api_key=KEY)
        self.math_generator_prompt = (
            "Write a math problem that required logic to solve."
        )

    def __call__(self, synapse: LogicSynapse) -> LogicSynapse:
        raw_logic_question, logic_question = self.get_challenge()
        synapse.logic_question = logic_question
        synapse.raw_logic_question = raw_logic_question
        return synapse

    def get_challenge(self) -> str:
        math_problem: str = self.get_math_problem()
        conditions: dict = self.get_condition()
        revised_math_question: str = self.get_revised_math_question(
            math_problem, conditions
        )
        return math_problem, revised_math_question

    def get_math_problem(self) -> str:
        messages = [
            {"role": "user", "message": self.math_generator_prompt},
        ]
        response = self.openai_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=128,
            temperature=0.5,
        )
        return response.choices[0].message["content"]

    def get_revised_math_question(self, math_problem: str, conditions: dict) -> str:
        messages = [
            {"role": "user", "message": self.get_math_problem},
            {"role": "system", "message": math_problem},
            {
                "role": "user",
                "message": "Please paraphrase by adding word or expression to this question as if you were a {profile} who is {mood} and write in a {tone} tone. You can use incorrect grammar, typo or add more context!".format(
                    **conditions
                ),
            },
        ]
        response = self.openai_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=256,
            temperature=0.5,
        )
        return response.choices[0].message["content"]

    def get_condition():
        profiles = [
            "math enthusiast",
            "math student",
            "research mathematician",
            "math teacher",
            "theoretical physicist",
            "engineer",
            "student",
            "teacher",
            "researcher",
            "physicist",
            "scientist",
            "mathematician",
            "data scientist",
            "math tutor",
            "math hobbyist",
            "data analyst",
            "data engineer",
            "data enthusiast",
            "data student",
            "data teacher",
            "data researcher",
        ]

        mood = [
            "curious",
            "puzzled",
            "eager",
            "analytical",
            "determined",
            "excited",
        ]

        tone = [
            "inquisitive",
            "thoughtful",
            "meticulous",
            "enthusiastic",
            "serious",
            "playful",
        ]

        return {
            "profile": random.choice(profiles),
            "mood": random.choice(mood),
            "tone": random.choice(tone),
        }
