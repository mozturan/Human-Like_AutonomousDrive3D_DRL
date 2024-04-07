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

    Reward is based on how close the car is to the center of the lane.
    Reward is also based speed maintenance using a gaussian function.
    Reward = (1-d) * exp(-(Target_speed - Current_speed)^2/(2*sigma^2))

    Args:
        max_cte (float): maximum cte at which the agent should stop
        target_speed (float): speed at which the agent should move
        sigma (float): standard deviation of the gaussian distribution
        action_cost (float): cost of taking an action
    
    Returns:
        float: reward

    """

    def __init__(self,max_cte, target_speed, sigma, action_cost):
        self.speed = target_speed
        self.max_cte = max_cte
        self.sigma = sigma
        self.action_cost = action_cost

    def __call__(self, state, action, info):
        return self.speed
    
    # A function preprocessing state, action and info
    def preprocess(self, state, action, info):
        pass

    # A function calculating the reward
    def calculate_reward(self, state, action, info):
        pass
    
