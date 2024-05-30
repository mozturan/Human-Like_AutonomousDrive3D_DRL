import gym
import gym_donkeycar
import numpy as np
import time
import wandb
import random

from src.environment.wrapper import Roscoe
from src.utils.config_loader import load_config, CONFIG_PATH
from src.agents import sac

conf = load_config(CONFIG_PATH)
env = gym.make("donkey-generated-track-v0", conf=conf)
obs, reward, done, info = env.reset()

roscoe = Roscoe(state=obs,
             action = np.array([0.0, 0.0]),
             done = done,
             info = info,
             max_cte = conf["max_cte"],
             sigma = 1.0,
             action_cost = 0.1,
             target_speed = 1.0)

obs, reward = roscoe.reset(obs, np.array([0.0, 0.0]), 
                           done, info)

agent = sac.SAC(state_size=obs.shape, 
                action_size=2, hidden_size=256,
                min_size=100, batch_size=64,
                model_name="SAC_Wrapper_Demo",
                max_action=0.6, temperature=0.1)

wandb.init(
    # set the wandb project where this run will be logged

    project="batch_54",

    config={
            "architecture": "AE-MLP",
            "dataset": "Generated-Track-v0",
            "epochs": 100,
            "batch_size": 64,
            "alpha": 0.0003,
            "beta": 0.001,
            "network": 256_256,
            "min mem size": 100,
            "max action": 0.6,
            "temperature": 0.1,
            "tau": 0.005,
            "gamma": 0.99,
            "reward scale": 1.0,
            "minibatch size": 64,
            "max_cte": 4.0,
            "sigma": 1.0,
            "action_cost": 0.1,
            "target_speed": 1.0,
            "noise": False,
            "env": "donkey-generated-track-v0",
            "wrapper": "Roscoe*",
    }
)

score_history = []
for episode in range(5000):
        obs, reward, done, info = env.reset()
        obs, reward = roscoe.reset(obs, np.array([0.0, 0.0]), 
                     done, info)
        
        episode_reward = 0
        episode_len = 0

        while not done and episode_len < 400:
                action = agent.choose_action(obs)
                normalized_action = [action[0], (action[1] / 2.0)+0.1]
                new_obs, reward, done, new_info = env.step(np.array(normalized_action))
                new_obs, reward = roscoe.step(new_obs, action, 
                        done, new_info)

                # print("Speed: ", new_info["speed"])
                # print("Forward Vel: ", new_info["forward_vel"])
                # print("CTE: ", new_info["cte"])
                # print("Reward: ", reward)

                episode_reward += reward
                episode_len +=1

                agent.remember(obs, action, reward, 
                        new_obs, done)
                
                agent.train()
                obs = new_obs

        score_history.append(episode_reward)
        avg_score = np.mean(score_history)

        agent.tensorboard.update_stats(episode_reward=episode_reward,
                score_avg=avg_score,
                episode_len=episode_len)
        
        wandb.log({"episode_length": episode_len, 
                   "episode_reward": episode_reward, 
                   "score_avg": avg_score})
                
env.close()