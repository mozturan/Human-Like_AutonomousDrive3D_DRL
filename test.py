import gym
import numpy as np
import gym_donkeycar

from src.environment.wrapper import Horace as Wrapper
from src.utils.config_loader import load_config, CONFIG_PATH
from hp import *
import keras
import tensorflow as tf

horace = keras.saving.load_model("models/Horace/actor_[450]_Horace.keras", custom_objects=None, compile=True, safe_mode=True)

def chose_action (model, state):
        state = tf.convert_to_tensor([state])
        actions, _ = model.sample_normal(state, reparameterize=False)

        return actions[0]

#* Initialize environment
conf = load_config(CONFIG_PATH)
env = gym.make("donkey-generated-track-v0", conf=conf)
obs, reward, done, info = env.reset()
start_action = np.array([0.0, 0.0])

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
obs, reward, done = wrapper.reset(obs, start_action, 
                           done, info)


for episode in range(500):

        print("Episode :  {}".format(episode))

        #* Reset environment and the wrapper
        obs, reward, done, info = env.reset()
        obs, reward, done = wrapper.reset(obs, np.array([0.0, 0.0]), 
                     done, info)
        
        #* Start episode
        while not done:

                #* Get action from agent and normalize it
                action = chose_action(horace, obs)
                normalized_action = [action[0], (action[1] / 2.0)+0.1]

                #* Step through environment and process the step
                new_obs, reward, done, new_info = env.step(np.array(normalized_action))
                new_obs, reward, done = wrapper.step(new_obs, action, 
                        done, new_info)
                
                #* Update observation
                obs = new_obs

env.close()