
# BipedalWalker with TD3


## Overview

Twin Delayed DDPG (TD3) is implemented for training an agent to solve the BipedalWalker environment. In this directory we implement the method using components for the Actor and Critic neural networks, experience replay buffer and Gaussian noise generator.

The code is based on the original code present in kaggle, with several modifications to enhance functionality and performance.

## Modifications Done

The original code was also slighty modified for running only the required method, improved compatibility, reproducibility, and functionality. The following modifications were done:

* Added a mechanism to save episode rewards and lengths to a CSV file for further analysis and visualization.
* Set random seeds for reproducibility to ensure that results can be replicated.
* Configured PyTorch to use deterministic algorithms.
* Added observation and reward scaling wrappers to preprocess the environment's observations and rewards.
* Integrated plotting function to visualize the training progress.

## Setup

1. Clone the repository

  ```bash
    git clone https://github.com/Rahuly123/Bipedal-walker-reinforcement-learning.git
    cd Bipedal-walker-reinforcement-learning/BipedalWalker-TD3

  ```

2. Install the prerequisites mentioned in the main README file.

3. Open the Jupyter notebook Bipedal_TD3.ipynb and execute the cells to start training.



    
## Credit
https://www.kaggle.com/code/auxeno/continuous-control-with-td3-rl/notebook

