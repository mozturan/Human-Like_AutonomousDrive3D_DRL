import numpy as np
from abc import ABC, abstractmethod
import math
import os

class RewardWrapper(ABC):
    def __init__(self, max_cte, max_delta, sigma, action_cost, target_speed):
        self.reward = 0
        self.max_cte = max_cte
        self.max_delta = max_delta
        self.sigma = sigma
        self.action_cost = action_cost
        self.target_speed = target_speed

    @abstractmethod
    def _reward(self, action, info, done):
        pass

    @abstractmethod
    def _action_reward(self, action):
        pass

    @abstractmethod
    def _cte_reward(self, cte):
        pass

    @abstractmethod
    def _speed_reward(self, speed):
        pass

    def __str__(self):
        return str(self.get_name())

    @abstractmethod
    def get_name(self):
        pass


class SmoothDrivingReward(RewardWrapper):

    def __init__(self, max_cte, max_delta, sigma, 
                 action_cost, target_speed):
        super().__init__(max_cte, max_delta, sigma, 
                         action_cost, target_speed)

    def __call__ (self, action, info, done):
        self.reward = self._reward(action, info, done)
        return self.reward
    
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
    
    def _action_reward(self, action):

        if self.last_action is not None:
            max_delta = 0.8 - (-0.8)
            cost = np.mean((action- self.last_action)**2
                            / max_delta**2)
            cost = cost * self.action_cost

        else:
            cost = 0.0

        self.last_action = action
        return cost

    def _cte_reward(self, cte):
        return (1.0 - (abs(cte)/self.max_cte)**2)**0.5

    def _speed_reward(self, speed):
        return math.exp(-(self.target_speed - speed)**2/(2*self.sigma**2))

    def get_name(self):
        return "SmoothDrivingReward"
