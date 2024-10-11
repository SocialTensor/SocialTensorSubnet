# <h1><center>🧠 LogicNet - Subnet 🤖</center></h1>

## Introduction

### Description
Our goal is to develop an open-source AI model capable of complex mathematics and detailed data analysis, enhanced by incentivized human feedback for continuous improvement.

- 📚 [Albert Frontend App](https://albert.aitprotocol.ai/)
- 📊 [Miners/Validator Stats](https://stats.aitprotocol.ai)
- 📈 [Grafana Dashboard](https://grafana.bactensor.io/d/miner/metagraph-miner?orgId=1)
- 📚 [Learn more about LogicNet](https://tonylvh.notion.site/LogicNet_SN35-1b44e52d308f47e7983af25bff6df90e)
  - More about the roadmap
  - Info on our open-source specialized model
  - Custom model benchmarking against other models
  - RLHF feature video demo

### Key Features

- 🚀 **Advanced Computational Network**: Incentivizing miners to enhance computational resources for complex AI/ML tasks.
- 💰 **Incentive Mechanism**:

  **Updated Reward System:**

  - **Initial Score Calculation**:
    - Each miner's response is evaluated to calculate an initial score using a weighted sum:
      - `score = (0.2 * similarity_score) + (0.8 * correctness_score) - 0.1 * time_penalty`
        - **Similarity Score**: Calculated based on the cosine similarity between the miner's reasoning and the self-generated ground truth answer.
        - **Correctness Score**: Determined by an LLM that assesses whether the miner's answer is correct based on the question and ground truth.
        - **Time Penalty**: Derived from the processing time of the response relative to the specified timeout.

  - **Rank-Based Incentives**:
    - Miners are ranked in descending order based on their initial scores.
    - Incentive rewards are assigned using a cubic function based on the rank:
      - `incentive_reward = -1.038e-7 * rank³ + 6.214e-5 * rank² - 0.0129 * rank - 0.0118 + 1`
      - This function scales rewards non-linearly to emphasize higher ranks, encouraging miners to provide higher-quality responses.
    - **Reward Scaling**:
      - The cubic function adjusts rewards so that top-ranked miners receive significantly higher rewards than lower-ranked ones.
      - Negative initial scores result in an incentive reward of zero.

  - **Purpose of the New Incentive Mechanism**:
    - **Enhance Competition**: By differentiating rewards based on rank, miners are motivated to outperform others.
    - **Improve Quality**: The emphasis on correctness and similarity encourages miners to provide accurate and relevant answers.
    - **Address Flat Incentive Curve**: The non-linear reward distribution resolves issues where miners previously received similar rewards despite varying performance levels.

- 🌟 **Continuous Improvement**: Expanding the math problem sets and categories to cover a broader range of topics.

### Neurons Documentation
- 📖 [Validator](docs/VALIDATOR.md)
- 📖 [Miner](docs/MINER.md)