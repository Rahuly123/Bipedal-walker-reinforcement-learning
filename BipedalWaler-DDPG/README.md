
# BipedalWalker with DDPG


## Overview

Deep Deterministic Policy Gradients is implemented for training an agent to solve the BipedalWalker environment. In this directory we implement the method using components for the Actor and Critic neural networks.

The code is based on the original repository by hahas94, with several modifications to enhance functionality and performance.

## Modifications Done

The original code was also slighty modified for running only the required method, improved compatibility, reproducibility, and functionality. The following modifications were done:

* Added a mechanism to save episode rewards to a CSV file for further analysis and visualization.
* Commented and removed the functions and imports that weren't required to avoid increased computational cost and time.
* Implemented a simpler plotting code for visualization.
* Adjusted hyperparameters to fine-tune the performance of the DDPG agent.

## Setup

1. Clone the repository

  ```bash
    git clone https://github.com/Rahuly123/Bipedal-walker-reinforcement-learning.git
    cd Bipedal-walker-reinforcement-learning/BipedalWaler-DDPG

  ```

2. Install the prerequisites mentioned in the main README file.

3. Open the Jupyter notebook DDPG_agent.ipynb and execute the cells to start training.



    
## Credit
https://github.com/hahas94/control/tree/main

