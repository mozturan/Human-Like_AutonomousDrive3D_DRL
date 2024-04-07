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
    
#! I might want to change its name
class ConstantSpeedReward(AbstractRewardType):

    """
    Reward function that is based on keeping 
    the agent moving at a constant speed.
    To get maximum reward at targetted speed
    the function uses a gaussian distribution.

    Args:
        speed (float): speed at which the agent should move
        sigma (float): standard deviation of the gaussian distribution
        mu (float): mean of the gaussian distribution
        max_reward (float): maximum reward that can be obtained
        min_reward (float): minimum reward that can be obtained
    
    Returns:
        float: reward

    """

    def __init__(self, speed):
        self.speed = speed
    def __call__(self, state, action, info):
        return self.speed
    
