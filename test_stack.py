import os
import gym
import gym_donkeycar
import numpy as np
from src.environment.rewards import ConstantSpeedReward
from src.utils.config_loader import load_config, CONFIG_PATH
# from src.agents.ddqn import ddqn
from src.environment.observations import CameraStack
from src.agents import sac
import time

#TODO: BETTER REWARD FUNCTION
#TODO: BETTER STATE HANDLING BY PRE TRAININ THE CNN'S

START_ACTION = [0.0,0.0]
score_history = []
 
conf = load_config(CONFIG_PATH)

env = gym.make("donkey-generated-roads-v0", conf=conf)

time.sleep(5)

Reward = ConstantSpeedReward(max_cte=conf["max_cte"], 
                             target_speed=1, 
                             sigma=0.2, action_cost=0.0)

camera = CameraStack(stack_size=2, image_shape=(120, 160))

obs, reward, done, info = env.reset()
observation = camera.reset(obs)
observation = camera.get_observation()

agent = sac.SAC(state_size=observation.shape, action_size=2, 
                hidden_size=512,min_size=20, buffer_size=100000, 
                batch_size=16)

for episode in range(5000):
        obs, reward, done, info = env.reset()
        observation = camera.reset(obs)
        observation = camera.get_observation()

        episode_reward = 0
        episode_len = 0

        while not done:
                # action = env.action_space.sample() #! does this work?
                action = agent.choose_action(observation)
                action = [action[0]/2.0, action[1]/2.0]
                new_obs, reward, done, new_info = env.step(np.array(action))
                new_observation = camera(new_obs)
                new_observation = camera.get_observation()
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


