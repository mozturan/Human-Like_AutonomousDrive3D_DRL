import numpy as np
from abc import ABC, abstractmethod
"""
This file contains abstract reward class for different reward functions for the environment.
Core is the abstract reward class 
which takes in the state, action and info then returns a reward
based on the selected reward function.
"""

# Abstract reward class
class AbstractRewardType(ABC):
    @abstractmethod
    def __call__(self, state, action, info):
        pass

class GaussianReward(AbstractRewardType):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, state, action, info):
        return np.random.normal(self.mean, self.std)
    

