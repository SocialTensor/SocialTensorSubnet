# Challenge for Synthetic Request
import os
import openai
import random
from logicnet.protocol import LogicSynapse
import bittensor as bt
from .human_noise import get_condition
from .math_generator.topics import TOPICS as topics
import mathgenerator
from datasets import load_dataset

DATASET_WEIGHT = [40,10,10,10,10,10,10]

class LogicChallenger:
    def __init__(self, base_url: str, api_key: str, model: str, dataset_weight: list):
        bt.logging.info(
            f"Initializing Logic Challenger with model: {model}, base URL: {base_url}."
        )
        self.model = model
        self.openai_client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.dataset_weight = [float(weight) for weight in dataset_weight.split(',')]

    def __call__(self, synapse: LogicSynapse) -> LogicSynapse:
        self.get_challenge(synapse)
        return synapse

    def get_challenge(self, synapse: LogicSynapse):
        # Get an atom logic problem
        atom_logic_question, atom_logic_answer = self.get_atom_logic_problem()
        if atom_logic_question is None or atom_logic_answer is None:
            bt.logging.error("Unable to retrieve atom logic problem. Retrying...")
            atom_logic_question, atom_logic_answer = self.get_atom_logic_problem()

        # Revise the problem
        conditions: dict = get_condition()
        revised_logic_question: str = self.get_revised_logic_question(
            atom_logic_question, conditions
        )
        
        # Set the synapse with the atom problem
        synapse.raw_logic_question = atom_logic_question
        synapse.ground_truth_answer = str(atom_logic_answer).replace("$", "").strip()
        synapse.logic_question = revised_logic_question

    def get_atom_logic_problem(self) -> str:
        resources = ['mathgenerator', 'zebralogicbench-grid', 'zebralogicbench-mc', 'ultrainteract', 'gsm8k', 'mmlustem', 'satmath']
        if len(self.dataset_weight) == 7:
            selected_resource = random.choices(resources, weights=self.dataset_weight, k=1)[0]
        else:
            bt.logging.warning("Invalid dataset weight configuration provided. Using default weights.")
            selected_resource = random.choices(resources, weights=DATASET_WEIGHT, k=1)[0]

        try:
            # Select an atom question and answer from the Mathgenerator
            if selected_resource == 'mathgenerator':
                selected_topic = random.choice(topics)
                subtopic = selected_topic["subtopic"]
                topic = selected_topic["topic"]
                atom_question, atom_answer = eval(f"mathgenerator.{topic}.{subtopic}()")
                if atom_question is None or atom_answer is None:
                    raise ValueError("Failed to get atom logic problem")
                bt.logging.debug("Generating math problem using Mathgenerator.")
                subtopic = subtopic.replace("_", " ").capitalize()
                topic = topic.replace("_", " ").capitalize()
                atom_question = atom_question.replace("$", "").strip()
                atom_question = f"Find the solution of this math problem:\n---\nTopic: {topic}, Subtopic: {subtopic}.\n{atom_question}\n---\n"
            
            # Select an atom question and answer from the ZebraLogicBench grid_mode
            elif selected_resource == 'zebralogicbench-grid':
                ds_grid = load_dataset("allenai/ZebraLogicBench-private", "grid_mode", token=os.environ.get('HF_TOKEN'))
                bt.logging.debug("Generating problem using ZebraLogicBench (grid mode).")
                data_set_grid = ds_grid['test']
                bt.logging.info(f"Loaded ZebraLogicBench (grid_mode) dataset with {len(data_set_grid['puzzle'])} entries")
                random_index = random.randint(0, len(data_set_grid['puzzle']) - 1)
                puzzle = data_set_grid['puzzle'][random_index]
                answer = data_set_grid['solution'][random_index]
                atom_question = f"Find the solution of this problem:\n---\n{puzzle}\n---\n"
                atom_answer = answer
            
            # Select an atom question and answer from the ZebraLogicBench mc_mode
            elif selected_resource == 'zebralogicbench-mc':
                ds_mc = load_dataset("allenai/ZebraLogicBench-private", "mc_mode", token=os.environ.get('HF_TOKEN'))
                bt.logging.debug("Generating problem using ZebraLogicBench (multiple choice mode).")
                data_set_mc = ds_mc['test']
                bt.logging.info(f"Loaded ZebraLogicBench (mc_mode) dataset with {len(data_set_mc['puzzle'])} entries")
                random_index = random.randint(0, len(data_set_mc['puzzle']) - 1)
                puzzle = data_set_mc['puzzle'][random_index]
                question = data_set_mc['question'][random_index]
                answer = data_set_mc['answer'][random_index]
                atom_question = f"Find the solution of this problem:\n---\n{puzzle}\n---\n{question}\n---\n"
                atom_answer = answer

            # Select an atom question and answer from the UltraInteract
            elif selected_resource == 'ultrainteract':
                ds = load_dataset("openbmb/UltraInteract_sft")
                bt.logging.debug("Generating problem using UltraInteract dataset.")
                data_set = ds['train']
                bt.logging.info(f"Loaded UltraInteract dataset with {len(data_set['instruction'])} entries")
                random_index = random.randint(0, len(data_set['instruction']) - 1)
                instruction = data_set['instruction'][random_index]
                response = data_set['response'][random_index]
                atom_question = f"Find the solution of this instruction:\n---\n{instruction}\n---\n"
                atom_answer = response
            
            # Select an atom question and answer from the GSM8K
            elif selected_resource == 'gsm8k':
                ds = load_dataset("openai/gsm8k", "main")
                bt.logging.debug("Generating problem using GSM8K dataset.")
                data_set = ds['train']
                bt.logging.info(f"Loaded GSM8K dataset with {len(data_set['question'])} entries")
                random_index = random.randint(0, len(data_set['question']) - 1)
                question = data_set['question'][random_index]
                answer = data_set['answer'][random_index]
                atom_question = f"Find the solution of this question:\n---\n{question}\n---\n"
                atom_answer = answer

            # Select an atom question and answer from the MMLU-STEM
            elif selected_resource == 'mmlustem':
                ds = load_dataset("TIGER-Lab/MMLU-STEM")
                bt.logging.debug("Generating problem using MMLU-STEM dataset.")
                data_set = ds['test']
                bt.logging.info(f"Loaded MMLU-STEM dataset with {len(data_set['question'])} entries")
                random_index = random.randint(0, len(data_set['question']) - 1)
                question = data_set['question'][random_index]
                answer_id = data_set['answer'][random_index]
                answer_choice = data_set['choices'][random_index]
                atom_question = f"Find the solution of this question:\n---\n{question}\n---\n"
                atom_answer = answer_choice[answer_id]

            # Select an atom question and answer from the SAT Math
            elif selected_resource == 'satmath':
                ds = load_dataset("mcaleste/sat_multiple_choice_math_may_23")
                bt.logging.debug("Generating problem using SAT Math dataset.")
                data_set = ds['train']
                bt.logging.info(f"Loaded SAT Math dataset with {len(data_set['Question'])} entries")
                random_index = random.randint(0, len(data_set['Question']) - 1)
                question = data_set['Question'][random_index]
                possible_answers = data_set['Possible Answers'][random_index]
                answer_id = data_set['Answer'][random_index]
                atom_question = f"Find the solution of this question:\n---\n{question}\n---\n"
                atom_answer = self.get_answer_value(possible_answers, answer_id)

        except Exception as e:
            bt.logging.error(f"Error accessing dataset {selected_resource}: {e}. Attempting to load an alternative dataset.")
            # Retry with a different dataset
            return self.get_atom_logic_problem()  

        return atom_question, atom_answer

    def get_revised_logic_question(self, logic_question: str, conditions: dict) -> str:
        # prompt = "Please paraphrase by adding word or expression to this question as if you were a {profile} who is {mood} and write in a {tone} tone. You can use incorrect grammar, typo or add more context! Don't add your solution! Just say the revised version, you don't need to be polite.".format(
        #     **conditions
        # )
        
        prompt = (
            "As a {profile} who is feeling {mood}, please rephrase the following math problem "
            "in a {tone} tone. Write it as you would naturally ask the question. "
            "Do not include the solution or add unnecessary context."
        ).format(**conditions)
        
        
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
            {"role": "assistant", "content": logic_question},
            {"role": "user", "content": prompt},
        ]

        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=256,
            temperature=0.7,
        )
        
        response = response.choices[0].message.content.strip()
        return response
    

    def get_answer_value(self, possible_answers, answer):
        # Get the value of the answer from the possible answers
        options = possible_answers.split()
        for i, option in enumerate(options):
            if option.startswith(answer + ")"):
                return options[i + 1]
        return None  # Return None if the answer is not found