
# Bipedal-walker-reinforcement-learning


## Overview

This project implements and evaluates several reinforcement learning agents that control a bipedal robot in the BipedalWalker-v3 environment from OpenAI Gymnasium. 

In this task, a
robot with two legs attached to a hull must walk across randomly generated terrain. The robot begins each episode in a standing position, and the episode finishes once the
robot reaches the end of the platform or the hull touches the ground.

![alt text](https://github.com/Rahuly123/Bipedal-walker-reinforcement-learning/blob/main/BipedalWaler-DDPG/Readme/bipedal_walker.gif?raw=true)

## Methods Implemented

The following methods are implemented on the classic environment of the Bipedal Walker:

#### Deep Deterministic Policy Gradients (DDPG)
Deep Deterministic Policy Gradient (DDPG) is an  approach that can
handle continuous action spaces. It is an off-policy, actor-critic algorithm that uses neural networks to
represent the policy and action-value function.
#### Twin Delayed DDPG (TD3)
Twin Delayed DDPG (TD3), an extension to DDPG uses two critic networks, out
of which the algorithm uses the smallest value estimate, reducing the overestimation bias.
#### Advantage Actor Critic (A2C)
Advantage Actor-Critic (A2C) is a synchronous version of the A3C method that utilizes parallel environments to enhance learning stability and convergence speed.

#### Proximal Policy Optimization (PPO)

Proximal Policy Optimization (PPO) is a policy gradient algorithm that aims to find a balance between exploring new strategies and exploiting known ones by restricting updates to the policy.

## Prerequisites

1. Upgrade setuptools package

  ```bash
    !pip install wheel setuptools pip --upgrade
  ```

2. Install the following packages

  ```bash
    !pip install gymnasium
    !pip install swig
    !pip install gymnasium[box2d]
    !pip install numpy
    !pip install matplotlib
    !pip install seaborn
    !pip install scipy
    !pip install tensorflow==2.9.0
    !pip install torch
  ```




    
## Results
A2C algorithm demonstrated the most stable reward variability, while TD3 and PPO exhibited significant fluctuations in rewards. DDPG exhibited no learning.


![alt text](https://github.com/Rahuly123/Bipedal-walker-reinforcement-learning/blob/main/BipedalWaler-DDPG/Readme/Results%20table.png?raw=true )



![alt text](https://github.com/Rahuly123/Bipedal-walker-reinforcement-learning/blob/main/BipedalWaler-DDPG/Readme/Algorithm%20graphs.jpeg?raw=true)


