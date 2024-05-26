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

    def reset(self):
        pass
    
#! I might want to change its name
class ConstantSpeedReward(RewardType):
    MIN_THROTTLE = 0.0
    MAX_THROTTLE = 1.0
    MIN_STEERING = -1.0
    MAX_STEERING = 1.0
    STEERING_DIFF = 0.15
    SHAKE_REWARD_WEIGHT = 1.0
    THROTTLE_REWARD_WEIGHT = 5.0

    def __init__(self,max_cte, target_speed, sigma, action_cost):
        
        self.target_speed = target_speed
        self.max_cte = max_cte
        self.sigma = sigma
        self.action_cost = action_cost

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

        if done:
            return self._calculate_done_reward(throttle=action[1], done=done)
        if forward_vel < 0.0:
            return 10*forward_vel
  
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
    
    def _calculate_done_reward(self, throttle, done) -> float:
            norm_throttle = (abs(throttle) - self.MIN_THROTTLE) / (self.MAX_THROTTLE - self.MIN_THROTTLE)
            return (-10 - (norm_throttle * self.THROTTLE_REWARD_WEIGHT)) * done

class AdvancedReward(ConstantSpeedReward):
    
    #! These are the default values
    #! Might need another way of config
    
    MIN_THROTTLE = 0.0
    MAX_THROTTLE = 1.0
    MIN_STEERING = -1.0
    MAX_STEERING = 1.0
    STEERING_DIFF = 0.15
    SHAKE_REWARD_WEIGHT = 1.0
    THROTTLE_REWARD_WEIGHT = 5.0

    def __init__(self, max_cte, target_speed, sigma, action_cost):
        super().__init__(max_cte, target_speed, sigma, action_cost)
        self.last_steering = 0.
        self.last_throttle = 0.

    def _reward(self, action, info, done) -> float:
        (cte, speed, forward_vel) = self._preprocess(info)

        if done:
            return self._calculate_done_reward(throttle=action[1], done=done)
        if forward_vel < 0.0:
            return -10+forward_vel
  
        reward_cte = self._calculate_cte_reward(cte)
        reward_speed = self._calculate_speed_reward(speed)
        reward_action = self._calculate_action_reward(action)
        reward_shake = self._calculate_shake_reward(steering=action[0])

        self.last_steering = action[0]
        self.last_throttle = action[1]

        return (reward_cte * reward_speed) # - reward_action - reward_shake

    def _calculate_shake_reward(self, steering) -> float:

        steering_dif = (steering - self.last_steering) / (self.MAX_STEERING - self.MIN_STEERING)

        if abs(steering_dif) > self.STEERING_DIFF:
            err = abs(steering_dif) - self.STEERING_DIFF
            shake_reward =(err**2) * self.SHAKE_REWARD_WEIGHT

        else:
            shake_reward = 0.0
        return shake_reward * self.SHAKE_REWARD_WEIGHT
    
    def _calculate_done_reward(self, throttle, done) -> float:
            norm_throttle = (abs(throttle) - self.MIN_THROTTLE) / (self.MAX_THROTTLE - self.MIN_THROTTLE)
            return (-10 - (norm_throttle * self.THROTTLE_REWARD_WEIGHT)) * done
        
    def reset(self):
        self.last_steering = 0.
        self.last_throttle = 0.