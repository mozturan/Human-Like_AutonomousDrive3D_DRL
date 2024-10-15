import numpy as np
from abc import ABC, abstractmethod
from keras.models import model_from_json as load
import os
import math
from collections import deque
import numpy as np

class StateWrapper(ABC):
    """
    Abstract class for state wrapper
    """

    def __init__(self, env, ae_path = None) -> None:
        super().__init__()

    @abstractmethod
    def __call__ (self, observation, action, info):
        pass

    def _crop(self, observation):

        return observation[40:120,:,:]

    def _process(self, observation):

        observation = np.clip(observation, 0, 255)
        observation = observation.astype(np.uint8)
        return observation/255.0


    @abstractmethod
    def _reset(self) -> None:
        pass
    
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def step(self, action) -> None:
        pass

class ExtendedStateVector(StateWrapper):
    def __init__(self, ae_path = None, state_history = 3) -> None:
        super().__init__(ae_path)
        
        self.state_history = state_history
        self.state = deque(maxlen=self.state_history)
        self.encoder = self._ae_load(ae_path)

    def __call__(self, observation, action, done, info):
        
        self.step(observation, action, done, info)
        state = np.hstack(self.state)

        return state

    def _info(self, info, action) -> np.array:

        return np.array([info["speed"],
                         info["forward_vel"],
                         action[0], action[1]])

    def _encode (self, observation) -> np.array:
        obs = self._crop(observation)
        obs = self._process(obs)
        obs = self._reduct(obs)
        return obs[0]
    
    def _reduct(self, observation) -> np.array:
        observation = np.expand_dims(observation, axis=0)
        observation = self.encoder.predict(observation, verbose=0)
        return observation

    def _ae_load(self, model_folder = "models/encoder_tracks/"):
        encoder_file = os.path.join(model_folder) + "encoder_model.json"
        weights_file = os.path.join(model_folder) + "encoder_weights.h5"

        with open(encoder_file, "r") as json_file:
            encoder_json = json_file.read()
        encoder = load(encoder_json)
        encoder.load_weights(weights_file)

        return encoder
    
    def _reset(self, obs, action, done, info) -> None:
        self.state = deque(maxlen=self.state_history)
        for _ in range(self.state_history):
            self.step(obs, [0,0], done, info)

    def reset(self, observation, action, done, info) -> None:
        action =np.array([0,0])
        self._reset(observation, action, done, info)
        state = np.hstack(self.state)
        return state

    def step(self, observation, action, done, info):

        observation_ = self._encode(observation)
        info_ = self._info(info, action)

        current_state = np.concatenate((observation_, info_), axis=0)
        self.state.append(current_state)


