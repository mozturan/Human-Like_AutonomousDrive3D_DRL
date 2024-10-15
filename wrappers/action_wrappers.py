import numpy as np
from abc import ABC, abstractmethod
import math

class ActionWrapper(ABC):

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_name(self):
        pass

class ActionClipping(ActionWrapper):
    def __init__(self):
        pass

    def reset(self):
        pass
    
    def step(self, action):
        smoothed_action = [action[0], (action[1] / 2.0)+0.1]

        return smoothed_action

    def get_name(self):
        return "ActionClipping"

class ExponentialMovingAverage(ActionWrapper):
    def __init__(self, smoothing_coef = 0.3):
        
        self.smoothing_coef = smoothing_coef
        self.smoothed_action = None

    def reset(self):
        self.smoothed_action = None

    def step(self, action):

        if self.smoothed_action is None:
                self.smoothed_action = np.zeros_like(action)
        self.smoothed_action = \
            self.smoothed_action * self.smoothing_coef + action * (1 - self.smoothing_coef)        
        
        return self.smoothed_action
    
    def get_name(self):
        return "LowPassFilter"
    
class WeightedMovingAverage(ActionWrapper):

    def __init__(self, window_size = 5, weights = None):
        self.window_size = window_size
        self.window = []

        if weights is None:
            self.weights = [i for i in range(1, window_size+1)]
            # self.weights = np.array(self.weights)
        else:
            self.weights = weights

    def reset(self):
        self.window = []

    def __call__(self, action):

        self.window.append(action)

        if len(self.window) > self.window_size:
            self.window.pop(0)

        if len(self.window) < self.window_size:
            return action

        # print("WMA: ", np.average(self.window, axis=0, weights=self.weights))
        return np.average(self.window, axis=0, weights=self.weights)
    
    def get_name(self):
        return "WeightedMovingAverage"
    
class InertiaTH(ActionWrapper):

    def __init__(self, threshold = 0.1):
        self.threshold = threshold
        self.last_action = None
        self.last_action_index = None

    def reset(self):
        self.last_action = None
        self.last_action_index = None

    def step(self, action):

        if self.last_action is not None:
            if np.abs(action - self.last_action) > self.threshold:
                self.last_action = action
                return action
            else:
                return self.last_action
        else:
            self.last_action = action
            return action
        
    def get_name(self):
        return "InertiaTH"
        

