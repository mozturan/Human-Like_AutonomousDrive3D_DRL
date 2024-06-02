import gym
import gym_donkeycar
import numpy as np
import time
import wandb
import random

from src.environment.wrapper import Roscoe
from src.utils.config_loader import load_config, CONFIG_PATH
from src.agents import sac
from hp import *

conf = load_config(CONFIG_PATH)
env = gym.make("donkey-generated-track-v0", conf=conf)
obs, reward, done, info = env.reset()

roscoe = Roscoe(state=obs,
             action = np.array([0.0, 0.0]),
             done = done,
             info = info,
             max_cte = conf["max_cte"],
             sigma = SIGMA,
             action_cost = ACTION_COST,
             target_speed = TARGET_SPEED)

obs, reward = roscoe.reset(obs, np.array([0.0, 0.0]), 
                           done, info)

agent = sac.SAC(state_size=obs.shape, 
                action_size=ACTION_SIZE, 
                hidden_size=HIDDEN_SIZE,
                min_size=MIN_SIZE, 
                batch_size=BATCH_SIZE,
                model_name=MODEL_NAME,
                max_action=MAX_ACTION, 
                temperature=TEMPERATURE)

wandb.init(
    # set the wandb project where this run will be logged

    project="Batch Size",
    name = "Roscoe fixed",

    config={
            "architecture": "AE-MLP",
            "dataset": "Generated-Track-v0",
            "epochs": 100,
            "batch_size": BATCH_SIZE,
            "alpha": ALPHA,
            "beta": BETA,
            "network": NETWORK,
            "min mem size": MIN_MEM_SIZE,
            "max action": MAX_ACTION,
            "temperature": TEMPERATURE,
            "tau": TAU,
            "gamma": GAMMA,
            "reward scale": REWARD_SCALE,
            "minibatch size": BATCH_SIZE,
            "max_cte": conf["max_cte"],
            "sigma": SIGMA,
            "action_cost": ACTION_COST,
            "target_speed": TARGET_SPEED,
            "noise": NOISE,
            "env": ENV,
            "wrapper": WRAPPER,
            "cte": CTE
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