import os
import gym
import gym_donkeycar
import numpy as np
from src.environment.rewards import ConstantSpeedReward
from src.utils.config_loader import load_config, CONFIG_PATH
# from src.agents.ddqn import ddqn
from src.environment.observations import Camera as Kinematics
from src.agents import sac
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

START_ACTION = [0.0,0.0]
score_history = []

# "exe_path": "/home/o/Applications/DonkeySimLinux/donkey_sim.x86_64",

conf = load_config(CONFIG_PATH)

env = gym.make("donkey-generated-roads-v0", conf=conf)

Reward = ConstantSpeedReward(max_cte=conf["max_cte"], 
                             target_speed=1, 
                             sigma=1, action_cost=0.0)

kinematics = Kinematics()

obs, reward, done, info = env.reset()
# observation = kinematics(START_ACTION, info)
observation = kinematics.reset(obs)

agent = sac.SAC(state_size=observation.shape, 
                action_size=2, 
                hidden_size=256,min_size=1000)

for episode in range(5000):
        obs, reward, done, info = env.reset()
        observation = kinematics.reset(obs)
        Reward.reset()
        episode_reward = 0
        episode_len = 0

        while not done:
                # action = env.action_space.sample() #! does this work?
                action = agent.choose_action(observation)
                new_obs, reward, done, new_info = env.step(np.array(action))
                new_observation = kinematics(new_obs, action, new_info)
                new_info["speed"] = new_info["speed"]*30000
                # reward = Reward(action, new_info, done)
                episode_reward += reward
                episode_len +=1

                agent.remember(observation, action, reward, 
                        new_observation, done)
                
                agent.train()
                observation = new_observation

        score_history.append(episode_reward)
        avg_score = np.mean(score_history)

        agent.tensorboard.update_stats(episode_reward=episode_reward,
                score_avg=avg_score,
                episode_len=episode_len)
        
env.close()


