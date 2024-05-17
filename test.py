import os
import gym
import gym_donkeycar
import numpy as np
from src.environment.rewards import ConstantSpeedReward
from src.utils.config_loader import load_config, CONFIG_PATH
# from src.agents.ddqn import ddqn
from src.environment.observations import Camera
from src.agents import sac

START_ACTION = [0.0,0.0]
score_history = []
 
conf = load_config(CONFIG_PATH)

env = gym.make("donkey-generated-roads-v0", conf=conf)

Reward = ConstantSpeedReward(max_cte=conf["max_cte"], 
                             target_speed=2, 
                             sigma=3, action_cost=0.0)

camera = Camera()

obs, reward, done, info = env.reset()
observation = camera(obs)

agent = sac.SAC(state_size=observation.shape, action_size=2, hidden_size=512,min_size=100)

for episode in range(5000):
        obs, reward, done, info = env.reset()
        observation = camera(obs)

        episode_reward = 0
        episode_len = 0

        while not done:
                # action = env.action_space.sample() #! does this work?
                action = agent.choose_action(observation)
                new_obs, reward, done, new_info = env.step(np.array(action))
                new_observation = camera(new_obs)
                reward = Reward(action, new_info, done)

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
        
        print("Memory Count: " , agent.memory.mem_cntr)

env.close()




