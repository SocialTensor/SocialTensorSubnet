from logicnet.validator import LogicChallenger
from logicnet.protocol import LogicSynapse

synapse = LogicSynapse()
challenger = LogicChallenger()


for _ in range(5):
    challenger(synapse)
    print(synapse)
    print()
