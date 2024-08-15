import numpy as np
from abc import ABC, abstractmethod
import math

class ActionWrapper(ABC):

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def step(self, action):
        pass
    
    @abstractmethod
    def get_name(self):
        pass

class LowPassFilter(ActionWrapper):
    def __init__(self, smoothing_coef = 0.3):
        
        self.smoothing_coef = smoothing_coef
        self.smoothed_action = None

    def reset(self):
        self.smoothed_action = None

    def step(self, action, smooth=True):

        if smooth:
            if self.smoothed_action is None:
                self.smoothed_action = np.zeros_like(action)
            self.smoothed_action = self.smoothed_action * self.smoothing_coef + action * (1 - self.smoothing_coef)
        else:
            self.smoothed_action = [action[0], (action[1] / 2.0)+0.1]
        
        return self.smoothed_action


class MovingAverage(ActionWrapper):

    def __init__(self, window_size = 5):
        self.window_size = window_size
        self.window = []

    def reset(self):
        self.window = []

    def step(self, action):

        self.window.append(action)

        if len(self.window) > self.window_size:
            self.window.pop(0)

        return np.mean(self.window, axis=0)
    

class WeightedMovingAverage(ActionWrapper):

    def __init__(self, window_size = 5):
        self.window_size = window_size
        self.window = []

        self.weights = [i for i in range(1, window_size+1)]

    def reset(self):
        self.window = []

    def step(self, action):

        self.window.append(action)

        if len(self.window) > self.window_size:
            self.window.pop(0)

        return np.average(self.window, axis=0, weights=self.weights)
    
