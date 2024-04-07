import numpy as np
from abc import ABC, abstractmethod
import math

"""
This file contains abstract reward class for different reward functions for the environment.
Core is the abstract reward class 
which takes in the state, action and info then returns a reward
based on the selected reward function.
"""

# Abstract reward class
class RewardType(ABC):
    @abstractmethod
    def __call__(self, action, info, done):
        pass

    @abstractmethod
    def _preprocess(self, info):
        pass

    @abstractmethod
    def _reward(self, action, info, done):
        pass
    
#! I might want to change its name
class ConstantSpeedReward(RewardType):

    """
    Reward function class that is based on keeping 
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
        self.target_speed = target_speed
        self.max_cte = max_cte
        self.sigma = sigma
        self.action_cost = action_cost

    def __call__(self, action, info, done) -> float: 
        return self._reward(action, info, done)

    # A function preprocessing state, action and info
    def _preprocess(self, info) -> tuple:

        #TODO: This need to be handled better
        """
        ! State might not be necessary
        INFO :  {'pos': (0.0, 0.0, 0.0), 
                 'cte': 0.0, 
                 'speed': 0.0, 
                 'forward_vel': 0.0, 
                 'hit': 'none',
                 'gyro': (0.0, 0.0, 0.0), 
                 'accel': (0.0, 0.0, 0.0), 
                 'vel': (0.0, 0.0, 0.0), 
                 'lidar': [], 
                 'car': (0.0, 0.0, 0.0), 
                 'last_lap_time': 0.0, 
                 'lap_count': 0},

        action : [0.0, 0.0]
        done : False
        """

        pos = info["pos"]
        cte = info["cte"]
        speed = info["speed"]
        gyro = info["gyro"]

        return (cte, speed)

    # A function calculating the reward
    def _reward(self, action, info, done) -> float:

        (cte, speed) = self._preprocess(info)

        if done:
            return -1.0
        if cte > self.max_cte:
            return -1.0
        
        reward_cte = (1.0 - math.fabs((cte/self.max_cte)))
        reward_speed = math.exp(-(self.target_speed - speed)**2/(2*self.sigma**2))
        reward_action = self.action_cost * np.linalg.norm(action)

        return (reward_cte * reward_speed) - reward_action
    
