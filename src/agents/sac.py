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
import keras

tf.random.set_seed(43)

class CriticNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=128, fc2_dims=128,
            name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)

        q = self.q(action_value)

        return q

class ValueNetwork(keras.Model):
    def __init__(self, fc1_dims=128, fc2_dims=128,
            name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.v = Dense(1, activation=None)

    def call(self, state):
        state_value = self.fc1(state)
        state_value = self.fc2(state_value)

        v = self.v(state_value)

        return v

class ActorNetwork(keras.Model):

    def __init__(self, max_action, fc1_dims=128,
            fc2_dims=128, n_actions=2, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action
        self.noise = 1e-6

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation=None)
        self.sigma = Dense(self.n_actions, activation=None)

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)
        # might want to come back and change this, perhaps tf plays more nicely with
        # a sigma of ~0
        sigma = tf.clip_by_value(sigma, self.noise, 1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.call(state)
        probabilities = tfp.distributions.Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.sample() # + something else if you want to implement
        else:
            actions = probabilities.sample()

        action = tf.math.tanh(actions)*self.max_action
        log_probs = probabilities.log_prob(actions)
        log_probs -= tf.math.log(1 - tf.math.pow(action,2)+self.noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

        return action, log_probs
    
class SAC:
    def __init__(self, state_size, action_size, alpha=0.0003, beta=0.001, hidden_size=512, temperature=0.03,
                 gamma=0.99, tau=0.005, buffer_size=int(1e6), batch_size=256, reward_scale=1.0, model_name = "SAC_DEMO"):
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
        
        self.model_name = f"{model_name}_{int(time.time())}"

        # Actor Network (w/ Target Network)
        self.actor_local = self._actor_network()
        self.actor_target = self._actor_network()
        self.actor_optimizer = Adam(lr=alpha)
        
        # Critic Network (w/ Target Network)
        self.critic_local = self._critic_network()
        self.critic_target = self._critic_network()
        self.critic_optimizer = Adam(lr=beta)
        
        self.hidden_size = hidden_size
        self.tempereture = temperature

        # Noise process
        self.noise = self.OUNoise(action_size)
        
        # Replay memory
        self.memory = ReplayBuffer(buffer_size, self.state_size,
                                   self.action_size, batch_size)
        
        # Save the hyperparameters
        self.gamma = gamma
        self.tau = tau
        
        self.actor = ActorNetwork(max_action=1.0, fc1_dims=hidden_size, fc2_dims=hidden_size,
                                  n_actions=action_size, name='actor')
        self.critic_1 = CriticNetwork(n_actions=action_size, fc1_dims=hidden_size,
                                      fc2_dims=hidden_size, name='critic_1')
        self.critic_2 = CriticNetwork(n_actions=action_size, fc1_dims=hidden_size,
                                      fc2_dims=hidden_size, name='critic_2')
        self.value = ValueNetwork(fc1_dims=hidden_size, fc2_dims=hidden_size, name='value')
        self.target_value = ValueNetwork(fc1_dims=hidden_size, fc2_dims=hidden_size, name='target_value')

        self.actor.compile(optimizer=self.actor_optimizer)
        self.critic_1.compile(optimizer=self.critic_optimizer)
        self.critic_2.compile(optimizer=self.critic_optimizer)
        self.value.compile(optimizer=self.critic_optimizer)
        self.target_value.compile(optimizer=self.critic_optimizer)
        self.scale = reward_scale

        self.update_network_parameters(tau=1)
        
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/sac/{self.model_name}")
        
    def choose_action(self, observation):
        """
        Chooses the next action based on the current observation
        """

        #* A noise added in sample_normal function
        state = tf.convert_to_tensor([observation])
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        
        return actions[0]

    def remember(self, state, action, reward, next_state, done):
        """
        Stores the next experience in the replay memory
        """
        self.memory.store_transition(state, action, reward, next_state, done)
    
    def update_network_parameters(self, tau=None):
        """
        Updates the parameters of the critic and actor networks given a mini-batch of experiences
        """

        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_value.weights
        for i, weight in enumerate(self.value.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_value.set_weights(weights)


