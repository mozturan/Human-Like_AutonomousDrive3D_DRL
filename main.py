import os
import gym
import gym_donkeycar
import numpy as np
from src.environment.rewards import ConstantSpeedReward
from src.utils.config_loader import load_config, CONFIG_PATH
from src.agents.ddqn import ddqn

conf = load_config(CONFIG_PATH)
env = gym.make("donkey-generated-roads-v0", conf=conf)

Reward = ConstantSpeedReward(max_cte=conf["max_cte"], 
                             target_speed=10, 
                             sigma=3, action_cost=0.1)

obs, reward, done, info = env.reset()
done= False
while not(done):
        # action = env.action_space.sample() #! does this work?
        action = [0.1,0.3]
        obs, reward, done, info = env.step(action)
        reward = Reward(action, info, done)
        print(info)



