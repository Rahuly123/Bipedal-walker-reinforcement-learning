
# BipedalWalker with A2C, Vectorized Environment


## Overview

Advantage Actor Critic (A2C) is implemented for training an agent to solve the BipedalWalker environment using a vectorized environment setup.
Solving this environment requires an average total reward of over 300 over 100 consecutive episodes. In this directory we solve the environment by using the A2C algorithm with Adam optimizer.

The code is based on the original repository by Rafael1s, with several modifications to enhance functionality and performance.
## Modifications Done

From the original repository, only files needed to execute A2C were extracted and run. The original code was also slighty modified for improved compatibility, reproducibility, and functionality. The following modifications were done:

* Updated the environment from 'BipedalWalker-v2' to 'BipedalWalker-v3'.
* Removed duplicate import statements.
* Added numpy seeding for reproducibility.
* Adjusted the training loop to log scores, save models periodically, and break the loop when the environment is solved.
* Enhanced logging and visualization by exporting scores to a CSV file and improving plotting.
* Added environment closure at the script's end to release resources.

## Setup

1. Clone the repository

  ```bash
    git clone https://github.com/Rahuly123/Bipedal-walker-reinforcement-learning.git
    cd Bipedal-walker-reinforcement-learning/BipedalWalker-A2C

  ```

2. Install the prerequisites mentioned in the main README file.

3. Open the Jupyter notebook BipedalWalker_A2C_VecEnv_Adam.ipynb and execute the cells to start training.



    
## Credit
https://github.com/Rafael1s/Deep-Reinforcement-Learning-Algorithms/tree/master/BipedalWalker-A2C-VectorizedEnv

