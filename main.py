import os
import gym
import gym_donkeycar
import numpy as np
from src.environment.rewards import ConstantSpeedReward
from src.utils.config_loader import load_config, CONFIG_PATH
# from src.agents.ddqn import ddqn
from src.environment.observations import Kinematics
from src.agents.ddqn import ddqn

START_ACTION = [0.0,0.0]
score_history = []

conf = load_config(CONFIG_PATH)

env = gym.make("donkey-generated-roads-v0", conf=conf)

Reward = ConstantSpeedReward(max_cte=conf["max_cte"], 
                             target_speed=3, 
                             sigma=3, action_cost=0.0)

kinematics = Kinematics()

obs, reward, done, info = env.reset()
observation = kinematics(START_ACTION, info)

agent = ddqn.DDQN(state_size=observation.shape, steering_container = 5, throttle_container=4)

for episode in range(5000):
        obs, reward, done, info = env.reset()
        observation = kinematics(START_ACTION, info)

        episode_reward = 0
        episode_len = 0

        while not done:
                # action = env.action_space.sample() #! does this work?
                action, action_index = agent.get_action(observation)
                new_obs, reward, done, new_info = env.step(action)
                new_observation = kinematics(action, new_info)
                reward = Reward(action, new_info, done)

                episode_reward += reward
                episode_len +=1

                agent.remember(observation, action_index, reward, 
                        new_observation, done)
                
                agent.train()
                observation = new_observation

        score_history.append(episode_reward)
        avg_score = np.mean(score_history)

        agent.tensorboard.update_stats(episode_reward=episode_reward,
                score_avg=avg_score,
                episode_len=episode_len,
                epsilon=agent.epsilon)
        
        print("Memory Count: " , agent.memory.mem_cntr)

env.close()


