
# BipedalWalker with PPO, Vectorized Environment


## Overview

Proximal Policy Optimization (PPO) is implemented for training an agent to solve the BipedalWalker environment using a vectorized environment setup.
Solving this environment require an average total reward of over 300 over 100 consecutive episodes. Training of BipedalWalker is very difficult to train BipedalWalker by PPO with one agent. In this directory we solve the environment by using PPO with multi-agent algorithm.

The code is based on the original repository by Rafael1s, with several modifications to enhance functionality and performance.
## Modifications Done

From the original repository, only files needed to execute PPO were extracted and run. The original code was also slighty modified for improved compatibility, reproducibility, and functionality. The following modifications were done:

* Updated the environment from 'BipedalWalker-v2' to 'BipedalWalker-v3'.
* Removed duplicate import statements.Refined device selection to use 'cpu' if CUDA is unavailable.
* Added numpy seeding for reproducibility.
* Updated the Policy model and ppo_agent to align with the new environment.
* Added a save function for the model's actor, critic, and critic linear state dictionaries.
* Adjusted the training loop to log scores, save models periodically, and break the loop when the environment is solved.
* Enhanced logging and visualization by exporting scores to a CSV file and improving plotting.
* Refined the play_VecEnv function to reset and render the environment for specified episodes, ensuring proper resource cleanup.
* Added environment closure at the script's end to release resources.
## Setup

1. Clone the repository

  ```bash
    git clone https://github.com/Rahuly123/Bipedal-walker-reinforcement-learning.git
    cd Bipedal-walker-reinforcement-learning/BipedalWalker-PPO

  ```

2. Install the prerequisites mentioned in the main README file.

3. Open the Jupyter notebook BipedalWalker_PPO.ipynb and execute the cells to start training.



    
## Credit
https://github.com/Rafael1s/Deep-Reinforcement-Learning-Algorithms/tree/master/BipedalWalker-PPO-VectorizedEnv

