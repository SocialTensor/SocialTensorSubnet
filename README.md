# <h1 align="center">🧠 LogicNet - Subnet 🤖</h1>

## Introduction

### Description
LogicNet is a pioneering decentralized AI subnet focused on developing an open-source, high-performance model proficient in solving complex mathematical problems and conducting detailed data analysis. As the system evolves, it leverages incentivized feedback loops—miners are rewarded for improving model outputs—ensuring continuous quality enhancement.

**Resources:**
- 📚 [LogicNet Website](https://logicnet.ai/)
- 📚 [Albert Frontend App](https://albert.aitprotocol.ai/)
- 📈 [Grafana Dashboard](https://grafana.bactensor.io/d/miner/metagraph-miner?orgId=1) (for network metrics and performance tracking)

**Neurons Documentation**
- 📖 [Validator](docs/VALIDATOR.md): Details on how to run a Validator
- 📖 [Miner](docs/MINER.md): Details on how to run a Miner

### Key Features

- 🚀 **Advanced Computational Network**:  
  Incentivizes miners (participants who submit solutions) to refine their answers, ensuring an ever-improving logical reasoning framework.

- 💰 **Updated Incentive Mechanism**:
  - **Initial Score Calculation**:  
    Each response is assigned a preliminary score:
    ``` 
    score = (0.2 * similarity_score) 
            + (0.8 * correctness_score) 
            - (0.1 * time_penalty)
    ```
    - **Similarity Score**: Evaluated via cosine similarity between the miner's reasoning steps and the validator's ground-truth reasoning.
    - **Correctness Score**: Determined by an LLM comparing the miner’s final answer to the expected solution.
    - **Time Penalty**: A small deduction for longer response times relative to a set timeout.

  - **Rank-Based Incentives**:  
    After scoring all responses:
    - Miners are ranked by their scores.
    - Rewards follow a cubic function based on rank:
      ``` 
      incentive_reward = [-1.038e-7 * (rank^3)] 
                         + [6.214e-5 * (rank^2)] 
                         - (0.0129 * rank) 
                         - 0.0118 + 1
      ```
    Higher-ranked miners earn disproportionately higher rewards, encouraging both precision and efficiency.

- 🌟 **Continuous Improvement**:  
  The system dynamically introduces more complex and diverse queries. Over time, model quality, dataset richness, and miner skill all steadily improve.
