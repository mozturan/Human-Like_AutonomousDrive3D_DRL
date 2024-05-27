import numpy as np
from abc import ABC, abstractmethod
from keras.models import model_from_json as load
import os
import math

class Wrapper(ABC):

    def __init__(self):
        pass

class Roscoe(Wrapper):
    def __init__(self, state, action, done, info, 
                 max_cte, sigma, action_cost,
                 target_speed):
        self.target_speed = target_speed
        self.encoder = self._ae_load()

        self.state_len = len(self.encode(state))
        self.info_len = len(self._info_edit(info, action))

        self.max_cte = max_cte
        self.sigma = sigma
        self.action_cost = action_cost

        self._reset()

        self.step(state, action, done, info)

    
    def _reset(self):
        self.last_state = np.zeros((self.state_len,),
                                    dtype=np.float32)
        self.last_info = np.zeros((self.info_len,),
                                   dtype=np.float32)

    def encode(self, state):
        state = self._crop(state)
        state = self._process(state)
        state = self._reduct(state)
        return state[0]

    def _crop(self, state):
        return state[40:120,:,:]

    def _process(self, state):
        state = np.clip(state, 0, 255)
        state = state.astype(np.uint8)
        return state/255.0

    def _reduct(self, state):
        state = np.expand_dims(state, axis=0)
        state = self.encoder.predict(state, verbose=0)
        return state

    def _ae_load(self, model_folder = "models/encoder_tracks/"):
        
        encoder_file = os.path.join(model_folder) + "encoder_model.json"
        weights_file = os.path.join(model_folder) + "encoder_weights.h5"

        with open(encoder_file, "r") as json_file:
            encoder_json = json_file.read()
        encoder = load(encoder_json)
        encoder.load_weights(weights_file)

        return encoder

    def _info_edit(self, info, action):
        return np.array([info["speed"],
                         info["forward_vel"],
                         action[0], action[1]])

    def reset(self, state, action, done, info):
        self._reset()
        return self.step(state, action, done, info)

    def step(self, state, action, done, info):
        reward = self._reward(action, info, done)

        state = self.encode(state)
        info = self._info_edit(info, action)

        observation = np.concatenate((state, 
                                      info,
                                      self.last_state,
                                      self.last_info), 
                                      axis=0)

        self.last_state = state
        self.lastinfo = info

        return observation, reward
    
    def _reward(self, action, info, done):

        if done:
            return -1.0
        if info["cte"] > self.max_cte:
            return -1.0
        if info["forward_vel"] < 0:
            return -1.0
        
        reward_cte = self._calculate_cte_reward(info["cte"])
        reward_speed = self._calculate_speed_reward(info["speed"])
        reward_action = self._calculate_action_reward(action)

        return (reward_cte * reward_speed) - reward_action
    def _calculate_cte_reward(self, cte) -> float:
        return (1.0 - (abs(cte)/self.max_cte)**2)
    
    def _calculate_action_reward(self, action) -> float:
        return self.action_cost * np.linalg.norm(action)
    
    def _calculate_speed_reward(self, speed) -> float:
        return math.exp(-(self.target_speed - speed)**2/(2*self.sigma**2))
