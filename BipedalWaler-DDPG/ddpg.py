"""
Deep Deterministic Policy Gradients
-----------------------------------
This file contains the training loop of a DDPG system, plus a
function for evaluating a model.
"""

import gym as gym
from gym.spaces import Box
import numpy as np
import tensorflow as tf
# import wandb
# from dotenv import dotenv_values

# import utils


def train(agent, target_agent, params):
    """
    Training a new agent model.

    Args:
        agent: tensorflow.keras.model.Model
            Agent model class instance.

        target_agent: tensorflow.keras.model.Model
            Agent model class instance.

        params: dict
            Dictionary of hyperparameters.

        model_name: str (default: {env_name}_{seed})
            A name to save the model to.

        log_data: bool
            Whether to log collected data to wandb.

    Returns:
        episode_returns, episode_lengths, critic_losses: list
            Lists of episode returns, episode lengths and critic losses.

    """
    env_name = params['env_name']
    env = gym.make(id=env_name)
    assert isinstance(env.observation_space, Box), "This example only works for envs with continuous state spaces."

    # seed = params['seed']
    obs_shape = params['obs_shape']
    n_actions = params['n_actions']
    action_low, action_high = params['action_bounds']
    loc, scale = params['random_process_parameters']
    n_steps = params['n_steps']
    evaluation_frequency = params['evaluation_frequency']
    learning_starts = params['learning_starts']
    # env.action_space.seed(seed)

    actor_weights, critic_weights = agent.get_weights()
    target_agent.set_weights(actor_weights, critic_weights, tau=1)

    # model_name = params['model_name']
    steps_played = 0
    episode_returns = []
    episode_lengths = []
    critic_losses = []
    best_return = float('-inf')
    n_solved_episodes = 0
    n_failed_episodes = 0
    reward_threshold = env.spec.reward_threshold if env.spec.reward_threshold else 1e6

    # if log_data:
    #     wandb_params = dotenv_values(".env")
    #     project_name = wandb_params['PROJECT_NAME']
    #     entity = wandb_params['ENTITY']
    #     _ = wandb.init(name=model_name, project=project_name, entity=entity, config=params)
        # raise NotImplementedError("Some parameters need to be defined. Create a hidden file named `.env`,
        # in the root of the directory, where the wandb related parameters are defined.")

    while steps_played < n_steps:
        state = env.reset()  # seed=seed
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            if steps_played < learning_starts:
                action = env.action_space.sample()
            else:
                action = agent.act(state.reshape(obs_shape)).numpy()
                noise = np.random.normal(loc=loc, scale=scale, size=(1, n_actions))
                action = (action + noise).clip(action_low, action_high).reshape((n_actions, ))

            next_state, reward, terminated, truncated = env.step(action)[:4]
            done = terminated or truncated

            steps_played += 1
            episode_length += 1
            episode_reward += reward

            agent.store_transition(state, action, reward, next_state, 1-int(terminated))
            state = next_state

            # --------------------------- Train agent ---------------------------
            if steps_played >= learning_starts:
                critic_loss, predicted_Q, Qs = agent.learn(target_agent=target_agent)
                # if log_data:
                #     wandb.log({'predicted_Q': predicted_Q,
                #                'Q': Qs,
                #                'critic loss': critic_loss})
                critic_losses.append(critic_loss)
                actor_weights, critic_weights = agent.get_weights()
                target_agent.set_weights(actor_weights, critic_weights, tau=None)

        # ------------------------- Data collection -------------------------
        er = round(episode_reward)
        episode_returns.append(er)
        episode_lengths.append(episode_length)
        n_episodes = len(episode_returns)
        n_solved_episodes += int(er >= reward_threshold and not truncated)
        n_failed_episodes += int(terminated and not truncated and er < reward_threshold)
        print(f"step:{steps_played+1:<10} episode:{n_episodes:<7} return:{er:<7} length:{episode_length}")

        # if log_data:
        #     wandb.log({'episode return': er,
        #                'episode length': episode_length,
        #                'solved episodes': n_solved_episodes,
        #                'failed episodes': n_failed_episodes}
        #               )

        # ---------------------- Evaluate target agent ----------------------
        # if n_episodes % evaluation_frequency == 0:
        #     evaluation_episode_rewards = evaluate(agent=target_agent, env_name=env_name,
        #                                           n_episodes=10, obs_shape=obs_shape, n_actions=n_actions, seed=seed)
        #     mean_return = np.mean(evaluation_episode_rewards).round()
        #     if log_data:
        #         wandb.log({'evaluation mean return': best_return})
        #     print(f"Evaluation mean return: {mean_return}")
        #     if mean_return > best_return:
        #         print(f"Best evaluation mean return. Saving target agent model parameters!")
        #         best_return = mean_return
        #         target_agent.save_model(model_name)

    env.close()

    return episode_returns, episode_lengths, critic_losses

# def evaluate(agent, env_name, n_episodes, obs_shape, n_actions, seed=2023, render=False, record=False):
#     """
#     Evaluates a trained agent for a number of episodes and returns statistics of these episodes.

#     Args:
#         agent: Agent object
#             An agent object.

#         env_name: str
#             Name of the environment.

#         n_episodes: int
#             Number of episodes to run the agent.

#         obs_shape: tuple
#             Shape of observation state.

#         n_actions: int
#             Size of actions array.

#         seed: int
#             An integer for seeding the random environment behaviour.

#         render: bool (default: False)
#             Whether to render the evaluation.

#         record: bool (default: False)
#             Whether to record a video of the evaluation.

#     Returns:
#         episode_rewards: list
#             A list containing the episodic rewards.
#     """
#     render_mode = 'human' if render else 'rgb_array' if record else None
#     env = gym.make(id=env_name, render_mode=render_mode)
#     action_low, action_high = env.action_space.low, env.action_space.high
#     episode_returns = []

#     if record:
#         gif_images = []

#     for episode in range(n_episodes):
#         state, info = env.reset()  # seed=seed
#         done = False
#         episode_return = 0

#         while not done:
#             if record:
#                 image = env.render()
#                 gif_images.append(image)

#             action = agent.act(state.reshape(obs_shape)).numpy().clip(action_low, action_high).reshape((n_actions, ))
#             state, reward, terminated, truncated, info = env.step(action)
#             done = terminated or truncated
#             episode_return += reward

#             if render:
#                 env.render()

#         episode_returns.append(episode_return)

#     env.close()

#     if record:
#         utils.record_gif(images=gif_images, name=env_name)

#     return episode_returns

# =============== END OF FILE ===============
