from src.agents.ddqn.buffer import PER
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

tf.random.set_seed(43)


class DDQN:
    def __init__(self, state_size, steering_container, throttle_container, hidden_size=512, model_name = "DDQN_DEMO",
                 batch_size=256, memory_capacity=10000, min_mem_size=300, replace_target = 100,
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.999,
                 learning_rate=0.001, double_dqn=True, dueling=False):
        
        self.state_size = state_size #* input shape

        self.model_name = f"{model_name}_{int(time.time())}"
        (self.discrete_action_space, 
         self.action_space, self.n_actions) = self.process_action_space(
                                            steering_container, throttle_container)
        self.replace_target = replace_target
        self.mem_size = memory_capacity
        self.min_mem_size = min_mem_size
        self.hidden_size = hidden_size #* for experimenting
        self.batch_size = batch_size
        self.memory = PER(max_size=self.mem_size, min_size=self.min_mem_size,
                          input_shape=self.state_size, n_actions=self.n_actions, 
                          discrete=True)
        
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.double_dqn = double_dqn
        self.dueling = dueling
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/ddqn/{self.model_name}")
        
        self.optimizer = Adam(learning_rate==self.learning_rate)

        self.q_eval = self._build_model()
        self.q_target = self._build_model()
        self.update_network_parameters()

    def process_action_space(self, steering_container, throttle_container):

        # discrete action space is an array for action values
        # action space is an array for action indices

        steering = np.linspace(-0.5, 0.5,steering_container)
        throttle = np.linspace(-0.1, 0.5,throttle_container)

        grid1, grid2 = np.meshgrid(steering, throttle)
        discrete_action_space = np.column_stack((grid1.ravel(), grid2.ravel()))
    
        n_actions = len(discrete_action_space)
        action_space = [i for i in range(n_actions)]

        return discrete_action_space, action_space, n_actions

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def epsilon_dec(self):
        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)

    def update_network_parameters(self):
        self.q_target.set_weights(self.q_eval.get_weights())

    def get_action(self, state):

        if np.random.rand() <= self.epsilon:
            action_index = np.random.randint(0, self.n_actions)
            action = self.discrete_action_space[action_index]

        else:
            observation = np.expand_dims(state, axis=0)
            qs_ = self.q_eval.predict(observation, verbose=0)

            action_index = np.argmax(qs_)
            action = self.discrete_action_space[action_index]

        return action, action_index

    def train(self, terminal=False):

        if self.memory.mem_cntr < self.min_mem_size:
            return
        
        #? What is this?
        if self.epsilon > self.epsilon_end and self.memory.mem_cntr > self.batch_size:
            self.epsilon_dec()

        batch = self.memory.sample_buffer(self.batch_size)
        state, action, reward, new_state, done, sample_indices = batch


        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)

        #* get the q values of current states by main network
        q_pred = self.q_eval.predict(state, verbose=0) #targets

        #! for abs error
        target_old = np.array(q_pred)

        #* get the q values of next states by target network
        q_next = self.q_target.predict(new_state, verbose=0) 

        #* get the q values of next states by main network
        q_eval = self.q_eval.predict(new_state, verbose=0) # type: ignore #! target_next

        #* get the actions with highest q values
        max_actions = np.argmax(q_eval, axis=1)

        #* we will update this dont worry
        q_target = q_pred

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        #* new_q = reward + DISCOUNT * max_future_q
        q_target[batch_index, action_indices] = reward + \
                    self.gamma*q_next[batch_index, max_actions.astype(int)]*done

        #* error
        error = target_old[batch_index, action_indices]-q_target[batch_index, action_indices]
        self.memory.set_priorities(sample_indices, error)

        loss = self.q_eval.fit(state, q_target, verbose = 0, epochs=1,
                            batch_size=self.batch_size, callbacks=[self.tensorboard]) # type: ignore
        
        if self.memory.mem_cntr % self.replace_target == 0:
            self.update_network_parameters()
            print("Target Updated")


        return loss
    
    def save_model(self):
        self.model.save_weights(os.path.join("models", self.model_dir))

    def load_model(self, file_name):
        self.model.load_weights(file_name)

    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_size, activation='relu', input_shape=self.state_size))
        model.add(Dense(self.hidden_size, activation='relu'))
        model.add(Dense(self.n_actions))
        model.compile(loss='mse', optimizer=self.optimizer)
        return model
