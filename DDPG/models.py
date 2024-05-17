"""
Models
------
This file contains classes for defining agent models such as the
action-value network (Critic) and the agent policy (Actor) models.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

from memory import ReplayBuffer

class Actor(keras.Model):
    def __init__(self, params):
        """
        Implements the agent policy (actor), that can be used to sample actions.

        Args:
            params: dict
                Hyperparameters dictionary
        """
        super(Actor, self).__init__()

        n_actions = params['n_actions']
        fc1_neurons, fc2_neurons = params['n_neurons']
        lr = params['actor_lr']
        self.action_bounds = params['action_bounds']

        self.fc1 = keras.layers.Dense(units=fc1_neurons, activation='relu')

        self.fc2 = keras.layers.Dense(units=fc2_neurons, activation='relu',)

        self.output_layer = keras.layers.Dense( units=n_actions, activation='tanh',
                                                kernel_initializer=keras.initializers.
                                                RandomUniform(minval=-0.003, maxval=0.003)
                                                )

        self.optimizer = keras.optimizers.Adam(learning_rate=lr)

    def call(self, inputs):
        """
        Compute action(s) for given observation(s)
        Args:
            inputs: numpy.ndarray
                One or a minibatch of observation arrays.

        Returns:
            action: tf.Tensor
                The predicted action(s) value(s).
        """
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.output_layer(x)

        action = x * self.action_bounds[1]

        return action

    @tf.function
    def optimize(self, actor_gradients):
        """
        Update weight parameters of the model given computed gradients.

        Args:
            actor_gradients: list
                Gradients of the actor model.

        Returns:
            None
        """
        self.optimizer.apply_gradients(zip(actor_gradients, self.trainable_variables))


class Critic(keras.Model):
    def __init__(self, params):
        """
        Implements the agent Q-values (critic) model, that can be used to compute actions values.

        Args:
            params: dict
                Hyperparameters dictionary
        """
        super(Critic, self).__init__()

        fc1_neurons, fc2_neurons = params['n_neurons']
        lr = params['critic_lr']

        self.fc1 = keras.layers.Dense(units=fc1_neurons, activation="relu")
        self.fc2 = keras.layers.Dense(units=fc2_neurons, activation="relu",)
        self.out = keras.layers.Dense(units=1, kernel_initializer=keras.initializers.
                                      RandomUniform(minval=-3 * 10 ** -3, maxval=3 * 10 ** -3),
                                      )
        self.concat = keras.layers.Concatenate()

        self.optimizer = keras.optimizers.Adam(learning_rate=lr)

    def call(self, inputs):
        """
        Compute action value(s) for given [observation(s), actions(s)]
        Args:
            inputs: list
                A pair of one or a minibatch of states and actions, both of which are tf.Tensor.
        Returns:
            q_values: tf.Tensor
                The predicted state-action(s) value(s).
        """
        x = self.concat(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        q_values = self.out(x)

        return q_values

    @tf.function
    def optimize(self, critic_gradients):
        """
        Update weight parameters of the model given computed gradients.

        Args:
            critic_gradients: list
                Gradients of the critic model.

        Returns:
            None
        """
        self.optimizer.apply_gradients(zip(critic_gradients, self.trainable_variables))


class Agent:
    def __init__(self, params):
        """
        Defining agent class, that has both memory buffer, actor and critic models.

        Args:
            params: : dict
                Hyperparameters dictionary
        """
        self.params = params
        self.gamma = params['gamma']
        self.tau = params['tau']

        self.memory = ReplayBuffer(obs_shape=params['obs_shape'], n_actions=params['n_actions'],
                                   buffer_size=params['buffer_size'], minibatch_size=params['minibatch'])
        self.actor = Actor(params)
        self.critic = Critic(params)

    def act(self, state):
        """
        Predict action for given state(s).

        Args:
            state: numpy.ndarray or tf.Tensor
                One or a minibatch of states.

        Returns:
            action: tf.Tensor
                Predicted action(s).

        """
        action = self.actor.call(state)
        return action

    def get_q_values(self, state, action):
        """
        Predicts the action value for given state-action pair(s).

        Args:
            state: tf.Tensor
                One or a minibatch of states.

            action: tf.Tensor
                One or a minibatch of actions.

        Returns:
            q_values: tf.Tensor
                The action values.
        """
        q_values = self.critic.call(inputs=[state, action])
        return q_values

    def store_transition(self, state, action, reward, next_state, terminal):
        """
        Adds a new experienced transition to the agent memory buffer.

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
        self.memory.store_transition(state, action, reward, next_state, terminal)

    def get_weights(self):
        """
        Retrieve and return both actor and critic models learnable weights.

        Returns:
        weight_actor, weight_critic: list
            Weights of all layers in a list. Weights of each layer is a numpy.ndarray.
        """
        actor_weights = self.actor.get_weights()
        critic_weights = self.critic.get_weights()

        return actor_weights, critic_weights

    @tf.function
    def _update_weights(self, weights, target_weights, tau):
        result = [0 for _ in range(len(weights))]

        for i in range(len(weights)):
            result[i] = tau*weights[i] + (1-tau)*target_weights[i]

        return result

    def set_weights(self, actor_weights, critic_weights, tau=None):
        """
        Update weights of both actor and critic using the soft update rule.
        E.g. theta = tau*actor_weights + (1-tau)*theta.

        Args:
            actor_weights: list
                Actor model trainable weights, which is a list of numpy.ndarrays.

            critic_weights: list
                Critic model trainable weights, which is a list of numpy.ndarrays.

            tau: float (default: None)
                The soft update rule prarameter.

        Returns:
            None
        """
        if tau is None:
            tau = self.tau

        target_actor_weights = self.actor.get_weights()
        target_actor_weights = self._update_weights(actor_weights, target_actor_weights, tau)
        self.actor.set_weights(target_actor_weights)

        target_critic_weights = self.critic.get_weights()
        target_critic_weights = self._update_weights(critic_weights, target_critic_weights, tau)
        self.critic.set_weights(target_critic_weights)

        return

    @tf.function
    def _learn(self, states, actions, rewards, next_states, terminals, target_agent):

        # --------- Update the Critic model ---------

        mse = keras.losses.MeanSquaredError()

        with tf.GradientTape() as critic_tape:
            target_actions = target_agent.act(next_states)
            target_q_values = target_agent.get_q_values(next_states, target_actions)
            y = rewards + self.gamma * terminals * target_q_values
            Qs = self.get_q_values(states, actions)
            critic_loss = mse(y, Qs)
        critic_gradients = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimize(critic_gradients)

        # --------- Update the Actor model ---------
        with tf.GradientTape() as tape:
            actions = self.act(states)
            # multiply by -1 to minimize the negative value, same as maximizing it
            q_values = -1 * tf.reduce_mean(self.get_q_values(state=states, action=actions))
        dq_dtheta = tape.gradient(q_values, self.actor.trainable_variables)
        self.actor.optimize(dq_dtheta)

        return critic_loss, y, Qs

    def learn(self, target_agent):
        """
        Perform optimization step to update both the actor and critic model parameters.

        Args:
            target_agent: Agent object
                Target agent.

        Returns:
            loss: numpy.ndarray
                The q-value loss between agent and target models on minibatch of samples.

        """
        states, actions, rewards, next_states, terminals = self.memory.sample()
        loss, y, Qs = self._learn(states, actions, rewards, next_states, terminals, target_agent)
        loss = loss.numpy()
        y = tf.reduce_mean(y).numpy()
        Qs = tf.reduce_mean(Qs).numpy()
        return loss, y, Qs

    # def save_model(self, model_name):
    #     """
    #     Save the model weights to a file in the ./trained_models forlder.
    #     Both actor and critic weights are stored separately into the folder.

    #     Args:
    #         model_name: str
    #             Name to save model to.

    #     Returns:
    #         None
    #     """
    #     filepath = f'trained_models/{model_name}'

    #     self.actor.save_weights(filepath+"/actor_weights/")
    #     self.critic.save_weights(filepath + "/critic_weights/")

    # def load_model(self, model_name):
    #     """
    #     Load a saved model from the `./trained_models`.
    #     Returns:
    #         None

    #     """
    #     filepath = f"trained_models/{model_name}"
    #     self.actor.load_weights(filepath+"/actor_weights/").expect_partial()
    #     self.critic.load_weights(filepath + "/critic_weights/").expect_partial()

# =============== END OF FILE ===============
