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
        return self._reward(action, info, done)

    @abstractmethod
    def _preprocess(self, info):
        pass

    @abstractmethod
    def _reward(self, action, info, done):
        pass
    
#! I might want to change its name
class ConstantSpeedReward(RewardType):

    def __init__(self,max_cte, target_speed, sigma, action_cost):

        if not isinstance(max_cte, float) or max_cte <= 0:
            raise ValueError("max_cte must be a positive float")
        if not isinstance(target_speed, float) or target_speed <= 0:
            raise ValueError("target_speed must be a positive float")
        if not isinstance(sigma, float) or sigma <= 0:
            raise ValueError("sigma must be a positive float")
        if not isinstance(action_cost, float) or action_cost <= 0:
            raise ValueError("action_cost must be a positive float")
        
        self.target_speed = target_speed
        self.max_cte = max_cte
        self.sigma = sigma
        self.action_cost = action_cost

    def _preprocess(self, info) -> tuple:
        expected_keys = ["cte", "speed"]
        expected_types = [float, float]

        for key, value in zip(expected_keys, expected_types):
            if key not in info or not isinstance(info[key], value):
                raise ValueError(f"Invalid info dictionary. Missing key '{key}' or incorrect type.")

        cte = info["cte"]
        speed = info["speed"]

        return (cte, speed)

    def _reward(self, action, info, done) -> float:
        (cte, speed) = self._preprocess(info)

        if done:
            return -1.0
        if cte > self.max_cte:
            return -1.0
  
        reward_cte = self._calculate_cte_reward(cte)
        reward_speed = self._calculate_speed_reward(speed)
        reward_action = self._calculate_action_reward(action)

        return (reward_cte * reward_speed) - reward_action

    def _calculate_cte_reward(self, cte) -> float:
        return (1.0 - abs((cte/self.max_cte)))

    def _calculate_speed_reward(self, speed) -> float:
        return math.exp(-(self.target_speed - speed)**2/(2*self.sigma**2))

    def _calculate_action_reward(self, action) -> float:
        return self.action_cost * np.linalg.norm(action)
    
