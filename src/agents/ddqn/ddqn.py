from agents.ddqn.buffer import PER
from utils.board import ModifiedTensorBoard
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
import numpy as np
import gc
import time

tf.random.set_seed(43)


class DDQN:
    def __init__(self, state_size, action_size, hidden_size=256, batch_size=64, memory_capacity=10000,
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 learning_rate=0.001, double_dqn=True, dueling=False, tensorboard_log=None):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.memory = PER(max_size=memory_capacity)
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.double_dqn = double_dqn
        self.dueling = dueling
        self.tensorboard_log = tensorboard_log

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        self.optimizer = Adam(lr=self.learning_rate)
        if self.tensorboard_log is not None:
            self.tensorboard = ModifiedTensorBoard(log_dir=self.tensorboard_log)

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def epsilon_decay(self):
        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)

    def update_network_parameters(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)

        if self.double_dqn:
            action_values = self.model.predict(state)
            return np.argmax(action_values[0])

        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def train(self, terminal=False):
        if self.epsilon > self.epsilon_end and self.memory.mem_cntr > self.batch_size:
            self.epsilon_decay()

        if self.memory.mem_cntr < self.batch_size:
            return

        batch = self.memory.sample_buffer(self.batch_size)
        states, actions, rewards, next_states, dones = batch

        targets = self.model.predict(states)
        if self.double_dqn:
            target_actions = np.argmax(self.target_model.predict(next_states), axis=1)
        else:
            target_actions = np.argmax(self.target_model.predict(next_states), axis=1)

        target_values = self.target_model.predict(next_states)
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * target_values[i][target_actions[i]]

        if self.tensorboard_log is not None:
            loss = self.model.fit(states, targets, verbose=0, callbacks=[self.tensorboard],
                                  epochs=1, batch_size=self.batch_size, shuffle=False)
        else:
            loss = self.model.fit(states, targets, verbose=0, epochs=1, batch_size=self.batch_size, shuffle=False)

    def save_model(self, file_name):
        self.model.save_weights(file_name)

    def load_model(self, file_name):
        self.model.load_weights(file_name)

    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_size, activation='relu', input_shape=(self.state_size,)))
        model.add(Dense(self.action_size))
        model.compile(loss='mse', optimizer=self.optimizer)
        return model
