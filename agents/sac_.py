from . import ExtendedReplayBuffer as ReplayBuffer 
from .noise import OrnsteinUhlenbeckActionNoise as noise_OU
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers.legacy import Adam
import tensorflow as tf
import tensorflow_probability as tfp
import keras.backend as K
import gc
import numpy as np
import os
import time
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

        #! This part is for reparameterization trick
        #TODO: Check for reparameterization tricks

        if reparameterize:
            actions = probabilities.sample() # + something else if you want to implement
        else:
            actions = probabilities.sample()

        action = tf.math.tanh(actions)*self.max_action
        log_probs = probabilities.log_prob(actions)
        log_probs -= tf.math.log(1 - tf.math.pow(action,2)+self.noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

        return action, log_probs
    
class SAC_:
    def __init__(self, state_size, action_size, alpha=0.0003, 
                 beta=0.001, hidden_size=512, temperature=0.001,
                 gamma=0.99, tau=0.005, buffer_size=int(1e6), 
                 min_size=1000, batch_size=256, reward_scale=1.0, 
                 model_name = "SAC_DEMO", max_action = 1.0, use_noise = True):
        """
        * Params
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

        """
        # Actor Network (w/ Target Network)
        self.actor_local = self._actor_network()
        self.actor_target = self._actor_network()
        """
        self.actor_optimizer = Adam(learning_rate=alpha)
        """
        # Critic Network (w/ Target Network)
        self.critic_local = self._critic_network()
        self.critic_target = self._critic_network()
        """
        self.critic_optimizer = Adam(learning_rate=beta)

        self.noise_OU = noise_OU(mean=np.zeros(2), 
                                     std_deviation=float(0.08) * np.ones(2),
                                     noise=use_noise)

        self.hidden_size = hidden_size
        self.tempereture = temperature
        self.max_action = max_action
        # Noise process
        # self.noise = self.OUNoise(action_size)
        
        # Replay memory
        self.memory = ReplayBuffer(max_size=buffer_size, 
                                   input_shape=self.state_size, 
                                   n_actions=self.action_size, 
                                   discrete = False)
        self.min_size = min_size
        self.batch_size = batch_size
        
        # Save the hyperparameters
        self.gamma = gamma
        self.tau = tau
        
        self.actor = ActorNetwork(max_action=max_action, fc1_dims=hidden_size, fc2_dims=hidden_size,
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

        self.actor.build(input_shape=(None,self.state_size[0]))
        self.actor.summary()
        
        # self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/sac/{self.model_name}")
        
    def choose_action(self, observation, evaluate = False):
        """
        Chooses the next action based on the current observation
        """

        state = tf.convert_to_tensor([observation])
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        #* Add noise to action if not evaluating
        if not evaluate:
            noise = self.noise_OU()
            action = actions[0] + noise
            action = np.clip(action, -self.max_action, self.max_action)
            return action
        
        return actions[0]

    def remember(self, state, action, reward, next_state, done, _action):
        """
        Stores the next experience in the replay memory
        """
        self.memory.store_transition(state, action, reward, next_state, done, _action)
    
    def update_network_parameters(self, tau=None):
        """
        Updates the parameters of the critic and actor networks given a mini-batch of experiences
        """

        #* If no tau is provided, use default
        if tau is None:
            tau = self.tau

        #* Get the weights from the target network
        weights = []
        targets = self.target_value.weights

        #* Update the weights of the value network
        for i, weight in enumerate(self.value.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_value.set_weights(weights)

    def train(self):

        #* If not enough samples in the memory, don't train
        if self.memory.mem_cntr < self.min_size:
            return

        #* Sample from memory
        state, action, reward, new_state, done, _action = \
            self.memory.sample_buffer(self.batch_size)

        #* Convert to tensors
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        _actions = tf.convert_to_tensor(_action, dtype=tf.float32)

        #* Train the value network
        with tf.GradientTape() as tape:
            # target_actions, log_probs = self.actor.sample_normal(states_, reparameterize=False)
            # q1 = self.critic_1(states, target_actions)

            value = tf.squeeze(self.value(states),1)
            value_ = tf.squeeze(self.target_value(states_),1)

            current_policy_actions, log_probs = self.actor.sample_normal(states, reparameterize=False)

            log_probs = tf.squeeze(log_probs,1)
            q1_new_policy = self.critic_1(states, current_policy_actions)
            q2_new_policy = self.critic_2(states, current_policy_actions)

            critic_value = tf.squeeze(
                tf.math.minimum(q1_new_policy, q2_new_policy), 1
            )

            value_target = critic_value - (log_probs*self.tempereture) #* rescale the log probabilities

            value_loss = 0.5*keras.losses.MSE(value, value_target) #?! Why did i set this to 0.5?

        value_network_gradients = tape.gradient(value_loss, 
                                                self.value.trainable_variables)
        self.value.optimizer.apply_gradients(zip(value_network_gradients,
                                                self.value.trainable_variables))
        
        #* Train the actor
        with tf.GradientTape() as tape:
            #! in the original paper, they reparameterize here! Check for this
            new_policy_actions, log_probs = self.actor.sample_normal(states, reparameterize=True)

            #TODO: Implement loss function smothering values
            smoothness_penalty = tf.reduce_mean(tf.square(new_policy_actions - _actions)) #??a
            smoothness_weight = 0.3

            log_probs = tf.squeeze(log_probs,1)
            q1_new_policy = self.critic_1(states, new_policy_actions)
            q2_new_policy = self.critic_2(states, new_policy_actions)
            crtitic_value = tf.squeeze(
                tf.math.minimum(q1_new_policy, q2_new_policy), 1
            )

            actor_loss = (self.tempereture*log_probs) - crtitic_value
            actor_loss = tf.math.reduce_mean(actor_loss)
            actor_loss += smoothness_weight*smoothness_penalty #??b  

        actor_network_gradients = tape.gradient(actor_loss,
                                                self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradients,
                                                self.actor.trainable_variables))
        
        #* Train the critic
        with tf.GradientTape(persistent=True) as tape:
            # I didn't know that these context managers shared values?
            q_hat = self.scale*reward + self.gamma*value_*(1-done)
            q1_old_policy = tf.squeeze(self.critic_1(state, action), 1)
            q2_old_policy = tf.squeeze(self.critic_2(state, action), 1)
            critic_1_loss = 0.5 * keras.losses.MSE(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * keras.losses.MSE(q2_old_policy, q_hat)

        critic_1_network_gradient = tape.gradient(critic_1_loss,
                                        self.critic_1.trainable_variables)
        critic_2_network_gradient = tape.gradient(critic_2_loss,
            self.critic_2.trainable_variables)

        self.critic_1.optimizer.apply_gradients(zip(
            critic_1_network_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(
            critic_2_network_gradient, self.critic_2.trainable_variables))
        
        self.update_network_parameters()

        #* Delete garbage for better memory management
        gc.collect()
        K.clear_session()

        return actor_loss, critic_1_loss

    def save(self, episode, path):

        if not os.path.exists(path):
            os.makedirs(path)
        
        self.actor.save(f"{path}/_{episode}.keras")
        
