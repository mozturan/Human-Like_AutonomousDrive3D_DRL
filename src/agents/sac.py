from src.agents.buffer import ReplayBuffer
from src.utils.board import ModifiedTensorBoard
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
import numpy as np
import gc
import os
import time
import tensorflow_probability as tfp
import sys
import random as rndm

tf.random.set_seed(43)


class SAC:
    def __init__(self, state_size, action_size, alpha=0.0003, beta=0.001, hidden_size=512, temperature=0.03,
                 gamma=0.99, tau=0.005, buffer_size=int(1e6), batch_size=256, reward_scale=1.0):
        """
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            actor_lr (float): learning rate of the actor network
            critic_lr (float): learning rate of the critic network
            discount (float): discount factor
            tau (float): target network update rate
            buffer_size (int): replay buffer size
            batch_size (int): mini-batch size
        """

        self.state_size = state_size
        self.action_size = action_size
        
        # Actor Network (w/ Target Network)
        self.actor_local = self._actor_network()
        self.actor_target = self._actor_network()
        self.actor_optimizer = Adam(lr=alpha)
        
        # Critic Network (w/ Target Network)
        self.critic_local = self._critic_network()
        self.critic_target = self._critic_network()
        self.critic_optimizer = Adam(lr=beta)
        
        self.hidden_size = hidden_size
        self.reward_scale = reward_scale
        self.tempereture = temperature

        # Noise process
        self.noise = self.OUNoise(action_size)
        
        # Replay memory
        self.memory = ReplayBuffer(buffer_size, self.state_size,
                                   self.action_size, batch_size)
        
        # Save the hyperparameters
        self.gamma = gamma
        self.tau = tau
        
        # Set up the optimizer
        self.critic_criterion = tf.keras.losses.MeanSquaredError()
        self.actor_criterion = tf.keras.losses.MeanSquaredError()
        
        # Set up the epsilon greedy policy
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.min_epsilon = 0.01

    def OUNoise(self, action):
        pass

    def _actor_network(self):
        """
        Creates the actor network for the SAC agent
        """
        network = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='sigmoid')
        ])
        network.compile(optimizer=self.actor_optimizer, loss=self.actor_criterion)
        return network

    def _critic_network(self):
        """
        Creates the critic network for the SAC agent
        """
        network = Sequential([
            Dense(64, input_dim=self.state_size + self.action_size, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        network.compile(optimizer=self.critic_optimizer, loss=self.critic_criterion)
        return network
        
    def choose_action(self, observation):
        """
        Chooses the next action based on the current observation
        """
        self.actor_local.reset_states()
        state = np.reshape(observation, [1, self.state_size])
        action = self.actor_local.predict(state)[0]
        # Add noise to the action
        action = np.clip(action + self.noise.sample(), -1.0, 1.0)
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.min_epsilon, self.epsilon)
        return action

    def remember(self, state, action, reward, next_state, done):
        """
        Stores the next experience in the replay memory
        """
        self.memory.add(state, action, reward, next_state, done)
    
    def update_network_parameters(self, batch_size):
        """
        Updates the parameters of the critic and actor networks given a mini-batch of experiences
        """
        experiences = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = experiences
        
        # Update the critic network using the target actor network
        with tf.GradientTape() as tape:
            # Get the predicted next-state actions from the target actor network
            next_actions = self.actor_target(next_states)
            # Compute the target Q values using the target critic network
            Q_targets_next = self.critic_target([next_states, next_actions])
            # Compute the expected Q values
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
            # Compute the critic loss
            Q_expected = self.critic_local([states, actions])
            critic_loss = self.critic_criterion(Q_expected, Q_targets)
        
        # Backpropagate the critic loss
        critic_grad = tape.gradient(critic_loss, self.critic_local.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_local.trainable_variables))
        
        # Update the actor network
        with tf.GradientTape() as tape:
            # Compute the actor
            pass

