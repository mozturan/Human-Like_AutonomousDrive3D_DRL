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
    SHAKE_REWARD_WEIGHT = 1.0
    STEERING_DIFF = 0.15
    def __init__(self,max_cte, target_speed, sigma, action_cost):
        
        self.target_speed = target_speed
        self.max_cte = max_cte
        self.sigma = sigma
        self.action_cost = action_cost
        self.n_low_speed = 0
        self.min_speed = 0.3
        self.n_steps = 0
        self.last_action = 0.0

    def reset(self):
        self.n_low_speed = 0
        self.n_steps = 0
        self.last_action = 0.0

    def __call__(self, action, info, done):
        return self._reward(action, info, done)

    def _preprocess(self, info) -> tuple:
        expected_keys = ["cte", "speed", "forward_vel"]
        expected_types = [float, float, float]

        for key, value in zip(expected_keys, expected_types):
            if key not in info or not isinstance(info[key], value):
                raise ValueError(f"Invalid info dictionary. Missing key '{key}' or incorrect type.")

        cte = info["cte"]
        speed = info["speed"]
        forward_vel = info["forward_vel"]

        return (cte, speed, forward_vel)

    def _reward(self, action, info, done) -> float:
        (cte, speed, forward_vel) = self._preprocess(info)

        self.n_steps += 1
        if speed < self.min_speed and self.n_steps > 100:
            self.n_low_speed += 1
        else:
            self.n_low_speed = 0

        if self.n_low_speed > 40:
                done = True
        
        if done:
            return -1.0
        if cte > self.max_cte:
            return -1.0
        if forward_vel < 0:
            return -1.0
  
        reward_cte = self._calculate_cte_reward(cte)
        reward_speed = self._calculate_speed_reward(speed)
        # reward_action = self._calculate_action_reward(action)
        # reward_shake = self._calculate_continuity_reward(action[0])

        return (reward_cte * reward_speed) #- reward_action - reward_shake

    def _calculate_cte_reward(self, cte) -> float:
        return (1.0 - (abs(cte)/self.max_cte)**2)

    def _calculate_speed_reward(self, speed) -> float:
        return math.exp(-(self.target_speed - speed)**2/(2*self.sigma**2))
    
    def _calculate_action_reward(self, action) -> float:
        return self.action_cost * np.linalg.norm(action)
    
    def _calculate_continuity_reward(self, action) -> float:
        steering_diff = self.last_action - action
        self.last_action = action

        if abs(steering_diff) > self.STEERING_DIFF:
            err = abs(steering_diff) - self.STEERING_DIFF
            return (err**2) * self.SHAKE_REWARD_WEIGHT
        else:
            return 0.0    
