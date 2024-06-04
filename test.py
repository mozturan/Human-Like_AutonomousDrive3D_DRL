import gym
import numpy as np

from src.environment.wrapper import Gnod as Wrapper
from src.utils.config_loader import load_config, CONFIG_PATH
from hp import *
import keras.models as m
from src.agents.sac import ActorNetwork

agent = ActorNetwork(max_action=0.8, fc1_dims=256, fc2_dims=256,
                                  n_actions=2, name='actor')
# agent.compile(optimizer='adam', loss='mse')

architecture_file = f"models/actor/[0]_Gnod.json"
weights_file = f"models/actor/[0]_Gnod.h5"

with open(architecture_file, "r") as json_file:
        arch = json_file.read()
model = m.model_from_json(arch)
model.load_weights(weights_file)
model.summary()

#* Initialize environment
conf = load_config(CONFIG_PATH)
env = gym.make("donkey-generated-track-v0", conf=conf)
obs, reward, done, info = env.reset()

#* Initialize wrapper
wrapper = Wrapper(state=obs,
             action = np.array([0.0, 0.0]),
             done = done,
             info = info,
             max_cte = conf["max_cte"],
             sigma = SIGMA,
             action_cost = ACTION_COST,
             target_speed = TARGET_SPEED)

#* Reset the wrapper
obs, reward, done = wrapper.reset(obs, np.array([0.0, 0.0]), 
                           done, info)

evaluate = True
score_history = []
episode_lens = []

#* Start episode
for i in range(5):
        #* Reset environment and the wrapper
        obs, reward, done, info = env.reset()
        obs, reward, done = wrapper.reset(obs, np.array([0.0, 0.0]), 
                     done, info)
        
        #* Reset episode reward
        episode_reward = 0
        episode_len = 0

        while not done and episode_len < 400:

                #* Get action from agent and normalize it
                action = agent.sample_normal(obs)
                normalized_action = [action[0], (action[1] / 2.0)+0.1]

                new_obs, reward, done, new_info = env.step(np.array(normalized_action))
                new_obs, reward, done = wrapper.step(new_obs, action, 
                            done, new_info)

                #* Update episode reward
                episode_reward += reward
                episode_len +=1

                obs = new_obs

        score_history.append(episode_reward)
        episode_lens.append(episode_len)

env.close()

print("Mean Score: {}".format(np.mean(score_history)))
print("Mean Episode Length: {}".format(np.mean(episode_lens)))

for i in range(len(score_history)):
        print("Episode {} Reward {} Episode Length {}".format(i, score_history[i], episode_lens[i]))

