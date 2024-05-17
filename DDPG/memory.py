"""
Memory
------
This file contains class definition for the replay buffer used for storing
agent transitions which can be sampled in minibatches for training.
"""

import numpy as np
import tensorflow as tf

class ReplayBuffer:
    def __init__(self, obs_shape, n_actions, buffer_size, minibatch_size):
        """
        Implementing a replay buffer that cann be used to store and
        transitions and sample minibatches for learning.

        Args:
            obs_shape: tuple
                Shape of observations.
            n_actions: int
                Number of actions
            buffer_size: int
                Size of replay buffer.
            minibatch_size: int
                Minibatch size.
        """
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.buffer_size = buffer_size
        self.minibatch_size = minibatch_size

        self.states = np.zeros((self.buffer_size, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, 1, self.n_actions), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, 1, 1), dtype=np.float32)
        self.next_states = np.zeros((self.buffer_size, *self.obs_shape), dtype=np.float32)
        self.terminals = np.zeros((self.buffer_size, 1, 1), dtype=np.float32)

        self.counter = 0

    def store_transition(self, state, action, reward, next_state, terminal):
        """
        Adds a new experienced transition to the replay buffer.

        Args:
            state: numpy.ndarray
                Agent state at the current step.
            action: numpy.ndarray
                Performed action.
            reward: float
                Received reward.
            next_state: numpy.ndarray
                Agent state at the next step.
            terminal: int
                0 if agent step cause episode termination, else 1.

        Returns:
            None
        """
        index = self.counter % self.buffer_size

        self.states[index] = state,
        self.actions[index] = action,
        self.rewards[index] = reward,
        self.next_states[index] = next_state,
        self.terminals[index] = terminal

        self.counter += 1

        return

    def sample(self):
        """
        Samples a random minibatch of experiences transitions.

        Returns:
            states: tf.Tensor
                Agent states.
            actions: tf.Tensor
                Agent actions.
            rewards: tf.Tensor
                Agent rewards.
            next_states: tf.Tensor
                Agent next state.
            terminals: tf.Tensor
                Terminal state indicators.

        """
        index = min(self.buffer_size, self.counter)
        indices = np.random.choice(a=range(0, index), size=self.minibatch_size, replace=False)

        states = tf.convert_to_tensor(self.states[indices])
        actions = tf.convert_to_tensor(self.actions[indices])
        rewards = tf.convert_to_tensor(self.rewards[indices])
        next_states = tf.convert_to_tensor(self.next_states[indices])
        terminals = tf.convert_to_tensor(self.terminals[indices])

        return states, actions, rewards, next_states, terminals

# =============== END OF FILE ===============
