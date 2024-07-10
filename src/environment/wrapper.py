import numpy as np
from abc import ABC, abstractmethod
from keras.models import model_from_json as load
import os
import math

class Wrapper(ABC):

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

        #* Initialize last state and info with zeros
        self.last_state = np.zeros((self.state_len,),
                                    dtype=np.float32)
        self.last_info = np.zeros((self.info_len,),
                                   dtype=np.float32)
        
        self.last_action = None

    def encode(self, state):

        #* Crop state and convert to float32
        state = self._crop(state)
        state = self._process(state)
        state = self._reduct(state)
        return state[0]

    def _crop(self, state):

        #* Crop state
        return state[40:120,:,:]

    def _process(self, state):

        #* Convert to 0-255 just in case and normalize
        state = np.clip(state, 0, 255)
        state = state.astype(np.uint8)
        return state/255.0

    def _reduct(self, state):

        #* Encode state into a vector using encoder
        state = np.expand_dims(state, axis=0)
        state = self.encoder.predict(state, verbose=0)
        return state

    def _ae_load(self, model_folder = "models/encoder_tracks/"):
        
        #* Load encoder
        encoder_file = os.path.join(model_folder) + "encoder_model.json"
        weights_file = os.path.join(model_folder) + "encoder_weights.h5"

        with open(encoder_file, "r") as json_file:
            encoder_json = json_file.read()
        encoder = load(encoder_json)
        encoder.load_weights(weights_file)

        return encoder

    def _info_edit(self, info, action):

        #* Edit info with necessery information
        return np.array([info["speed"],
                         info["forward_vel"],
                         action[0], action[1]])

    def reset(self, state, action, done, info):

        self._reset()
        return self.step(state, action, done, info)

    def step(self, state, action, done, info):

        #* Calculate reward
        reward = self._reward(action, info, done)

        #* Encode state
        state = self.encode(state)
        info = self._info_edit(info, action)

        #* Concatenate state and info
        observation = np.concatenate((state, 
                                      info,
                                      self.last_state,
                                      self.last_info), 
                                      axis=0)

        #* Update last state and info
        self.last_state = state
        self.last_info = info

        return observation, reward, done

    def _reward(self, action, info, done):
        pass

    def _calculate_cte_reward(self, cte) -> float:
        return (1.0 - (abs(cte)/self.max_cte)**2)

    def _calculate_speed_reward(self, speed) -> float:
        return math.exp(-(self.target_speed - speed)**2/(2*self.sigma**2))

    def _calculate_action_reward(self, action):
        pass
    def get_name(self):
        pass

class Roscoe(Wrapper):

    name = "Roscoe"
    def __init__(self, state, action, done, info, max_cte, sigma, action_cost, target_speed):
        super().__init__(state, action, done, info, max_cte, sigma, action_cost, target_speed)
    
    def _reward(self, action, info, done):

        if done:
            return -1.0
        if info["cte"] > self.max_cte:
            return -1.0
        if info["forward_vel"] < 0:
            return -1.0
        
        reward_cte = self._calculate_cte_reward(info["cte"])
        reward_speed = self._calculate_speed_reward(info["speed"])
        reward_action = self._calculate_action_reward(np.array(action))

        return (reward_cte * reward_speed) - reward_action
    
    def _calculate_action_reward(self, action) -> float:
        return self.action_cost * np.linalg.norm(action)

    def get_name(self):
        return self.name

    # Implement the __str__ method to return the name
    def __str__(self):
        return str(self.get_name())
    
class Gnod(Wrapper):

    name = "Gnod"
    def __init__(self, state, action, done, info, max_cte, sigma, action_cost, target_speed):
        super().__init__(state, action, done, info, max_cte, sigma, action_cost, target_speed)

    def _calculate_action_reward(self, action) -> float:

        if self.last_action is not None:
            max_delta = 0.8 - (-0.8)
            cost = np.mean((action- self.last_action)**2
                            / max_delta**2)
            cost = cost * self.action_cost

        else:
            cost = 0.0

        self.last_action = action
        return cost
    
    def _reward(self, action, info, done):

        if done:
            return -1.0
        if info["cte"] > self.max_cte:
            return -1.0
        if info["forward_vel"] < 0:
            return -1.0
        
        reward_cte = self._calculate_cte_reward(info["cte"])
        reward_speed = self._calculate_speed_reward(info["speed"])
        reward_action = self._calculate_action_reward(np.array(action))

        return (reward_cte * reward_speed) - reward_action
    
    def get_name(self):
        return self.name

    # Implement the __str__ method to return the name
    def __str__(self):
        return str(self.get_name())

class Faith(Wrapper):

    name = "Faith"
    def __init__(self, state, action, done, info, max_cte, sigma, action_cost, target_speed):
        super().__init__(state, action, done, info, max_cte, sigma, action_cost, target_speed)

    def _calculate_action_reward(self, action) -> float:

        if self.last_action is not None:
            max_delta = 0.8 - (-0.8)
            cost = np.mean((action- self.last_action)**2
                            / max_delta**2)
            cost = cost * self.action_cost

        else:
            cost = 0.0

        self.last_action = action
        return cost

    def _reward(self, action, info, done):

        if done:
            return -1.0
        if info["cte"] > self.max_cte:
            return -1.0
        if info["forward_vel"] < 0:
            return -1.0
        
        reward_cte = self._calculate_cte_reward(info["cte"])
        reward_speed = self._calculate_speed_reward(info["speed"])
        reward_action = self._calculate_action_reward(np.array(action))

        return (reward_cte * reward_speed) / (reward_action + 1)
    
    def get_name(self):
        return self.name

    # Implement the __str__ method to return the name
    def __str__(self):
        return str(self.get_name())
    























"""
Below is for Lidar and state encoding (Phase 2)
"""

class Horace(Wrapper):
    """
    This is for Lidar and state encoding
    """
    name = "Horace"
    def __init__(self, state, action, done, info, 
                 max_cte, sigma, action_cost,
                 target_speed):
        self.target_speed = target_speed
        self.encoder = self._ae_load()
        self.lidar_encoder = self._lidarae_load()

        self.lidar_len = len(self._lidar(info["lidar"]))
        self.state_len = len(self.encode(state))
        self.info_len = len(self._info_edit(info, action))
        self.max_cte = max_cte
        self.sigma = sigma
        self.action_cost = action_cost

        self._reset()

        self.step(state, action, done, info)

    def _reset(self):

        #* Initialize last state and info with zeros
        self.last_state = np.zeros((self.state_len,),
                                    dtype=np.float32)
        self.last_info = np.zeros((self.info_len,),
                                   dtype=np.float32)
        
        self.last_action = None

        self.last_lidar = np.zeros((self.lidar_len,),
                                    dtype=np.float32)
        
        self.done = False
        self.cost = 0

    def encode(self, state):

        #* Crop state and convert to float32
        state = self._crop(state)
        state = self._process(state)
        state = self._reduct(state)
        return state[0]

    def _crop(self, state):

        #* Crop state
        return state[40:120,:,:]

    def _process(self, state):

        #* Convert to 0-255 just in case and normalize
        state = np.clip(state, 0, 255)
        state = state.astype(np.uint8)
        return state/255.0

    def _reduct(self, state):

        #* Encode state into a vector using encoder
        state = np.expand_dims(state, axis=0)
        state = self.encoder.predict(state, verbose=0)
        return state

    def _ae_load(self, model_folder = "models/encoder_tracks/"):
        
        #* Load encoder
        encoder_file = os.path.join(model_folder) + "encoder_model.json"
        weights_file = os.path.join(model_folder) + "encoder_weights.h5"

        with open(encoder_file, "r") as json_file:
            encoder_json = json_file.read()
        encoder = load(encoder_json)
        encoder.load_weights(weights_file)

        return encoder

    def _info_edit(self, info, action):

        #* Edit info with necessery information
        return np.array([info["speed"],
                         info["forward_vel"],
                         action[0], action[1]])

    def reset(self, state, action, done, info):

        self._reset()
        return self.step(state, action, done, info)

    def step(self, state, action, done, info):

        lid = info["lidar"].copy()  
        lid = self._lidar_normalize(lid)

        if lid.min() < 0.1:
            print("Lidar min punishment: ", lid.min())
            self.done = True
        else: self.done = done

        #* Calculate reward
        reward = self._reward(action, info, self.done)

        #* Encode state
        state = self.encode(state)
        lidar = self._lidar(info["lidar"])

        info = self._info_edit(info, action)

        #* Concatenate state and info
        observation = np.concatenate((state, 
                                      lidar,
                                      info,
                                      self.last_state,
                                      self.last_info,
                                      self.last_lidar), 
                                      axis=0)

        #* Update last state and info
        self.last_state = state
        self.last_info = info
        self.last_lidar = lidar

        return observation, reward, self.done
    def _calculate_action_reward(self, action) -> float:

        if self.last_action is not None:
            max_delta = 0.8 - (-0.8)
            cost = np.mean((action- self.last_action)**2
                            / max_delta**2)
            self.cost = cost * self.action_cost

        else:
            self.cost = 0.0

        self.last_action = action
        return self.cost
    
    def _reward(self, action, info, done):

        if done:
            return -1.0
        if info["cte"] > self.max_cte:
            return -1.0
        if info["forward_vel"] < 0:
            return -1.0
        
        reward_cte = self._calculate_cte_reward(info["cte"])
        reward_speed = self._calculate_speed_reward(info["speed"])
        reward_action = self._calculate_action_reward(np.array(action))

        return (reward_cte * reward_speed) - reward_action
    
    def get_name(self):
        return self.name

    # Implement the __str__ method to return the name
    def __str__(self):
        return str(self.get_name())

    def _calculate_cte_reward(self, cte) -> float:
        return (1.0 - (abs(cte)/self.max_cte)**2)

    def _calculate_speed_reward(self, speed) -> float:
        return math.exp(-(self.target_speed - speed)**2/(2*self.sigma**2))

    def _lidar_normalize(self, lidar):

        #* Normalize lidar
        normalized_lidar = lidar.copy()
        normalized_lidar[normalized_lidar < 0] = 20.0
        normalized_lidar /= 20.0

        return normalized_lidar
    
    def _lidarae_load(self, model_folder = "models/encoder_tracks/"):
        
        #* Load encoder
        encoder_file = os.path.join(model_folder) + "lidar_encoder.json"
        weights_file = os.path.join(model_folder) + "lidar_encoder.h5"

        with open(encoder_file, "r") as json_file:
            encoder_json = json_file.read()
        encoder = load(encoder_json)
        encoder.load_weights(weights_file)

        return encoder
    
    def _lidar(self, lidar):

        #* Encode lidar
        lidar = self._lidar_normalize(lidar)
        lidar = np.expand_dims(lidar, axis=0)
        lidar = self.lidar_encoder.predict(lidar, verbose=0)
        return np.reshape(lidar, (16,)) 


















































    # def __init__(self, state, action, done, info, max_cte, sigma, action_cost, target_speed):
    #     super().__init__(state, action, done, info, max_cte, sigma, action_cost, target_speed)
        
    # def _calculate_action_reward(self, action) -> float:

    #     smoothing_coeff = 0.3
    #     if self.last_action is None:
    #         self.last_action = np.zeros_like(action)

    #     self.smoothed_action = self.smoothing_coef * self.smoothed_action + (1 - self.smoothing_coef) * action
    
