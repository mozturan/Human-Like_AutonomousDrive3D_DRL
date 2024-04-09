import os
import gym
import gym_donkeycar
import numpy as np
from src.environment.rewards import ConstantSpeedReward
from src.utils.config_loader import load_config, CONFIG_PATH
from src.agents.ddqn import ddqn
from src.environment.observations import Kinematics

START_ACTION = [0.0,0.0]
conf = load_config(CONFIG_PATH)
env = gym.make("donkey-generated-roads-v0", conf=conf)

Reward = ConstantSpeedReward(max_cte=conf["max_cte"], 
                             target_speed=10, 
                             sigma=3, action_cost=0.1)

kinematics = Kinematics()

obs, reward, done, info = env.reset()
observation = kinematics(START_ACTION, info)

agent = ddqn.DDQN(state_size=observation.shape, steering_container=5, throttle_container=3)


for episode in range(100):
        obs, reward, done, info = env.reset()
        observation = kinematics(START_ACTION, info)

        done= False
        while not done:
                # action = env.action_space.sample() #! does this work?
                action, action_index = agent.get_action(observation)
                new_obs, reward, done, new_info = env.step(action)
                new_observation = kinematics(action, new_info)
                reward = Reward(action, new_info, done)


                agent.remember(observation, action_index, reward, 
                        new_observation, done)
                
                observation = new_observation
                print(reward)


env.close()


