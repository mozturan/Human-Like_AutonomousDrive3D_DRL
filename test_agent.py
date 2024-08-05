import gym
import numpy as np
import gym_donkeycar
import wandb

from hp import *
import keras
import tensorflow as tf

from src.utils.config_loader import load_config, CONFIG_PATH
from src.environment.wrapper import Roscoe as Wrapper
from src.environment.action_shaping import SmoothingAction
from src.utils.performance import PerformanceMSE as Performance

model_string= "models/Generation_0/Nothing/actor_[333]_Nothing.keras"

#* Load model
model = keras.saving.load_model(model_string, 
                                custom_objects=None, 
                                compile=True, 
                                safe_mode=True)

def chose_action (model, state):
        state = tf.convert_to_tensor([state])
        actions, _ = model.sample_normal(state, reparameterize=False)

        return actions[0]

#* Initialize environment
conf = load_config(CONFIG_PATH)
env = gym.make("donkey-generated-track-v0", conf=conf)
obs, reward, done, info = env.reset()
start_action = np.array([0.0, 0.0])
action_wrapper = SmoothingAction(smoothing_coef = 0.5)

#* Initialize wrapper
wrapper = Wrapper(state=obs,
             action = np.array([0.0, 0.0]),
             done = done,
             info = info,
             max_cte = conf["max_cte"],
             sigma = SIGMA,
             action_cost = ACTION_COST,
             target_speed = TARGET_SPEED)

#* Initialize performanse
performance = Performance(ref_speed=TARGET_SPEED)

#* Reset the wrapper
obs, reward, done = wrapper.reset(obs, start_action, 
                           done, info)

score_history = []
max_episode_length = 1000

for episode in range(50):
        action_wrapper.reset()

        #* Reset environment and the wrapper
        obs, reward, done, info = env.reset()
        obs, reward, done = wrapper.reset(obs, np.array([0.0, 0.0]), 
                     done, info)

        #* Reset episode reward
        episode_reward = 0
        episode_len = 0

        #* Reset performance
        performance.reset()

        while not done and episode_len < max_episode_length:

                #* Get action from agent and normalize it
                action = chose_action(model, obs)
                normalized_action = action_wrapper.step(action, smooth = USE_SMOOTHING)

                #* Step through environment and process the step

                new_obs, reward, done, new_info = env.step(np.array(normalized_action))
                
                new_obs, reward, done = wrapper.step(new_obs, action, 
                        done, new_info)
                
                #* Update episode reward
                episode_reward += reward
                episode_len +=1

                #* Update observation
                obs = new_obs

                #* Update performance
                performance(cte = new_info["cte"], speed = new_info["speed"], action = normalized_action)

        score_history.append(episode_reward)
        avg_score = np.mean(score_history)

        #* Log to wandb
        mean_error, cte_avg, speed_avg, avg_delta = performance.get_metrics()

        print("   Episode: ", episode,
              "   Score: ", episode_reward,
              "   CTE: ", cte_avg,
              "   Speed: ", speed_avg,
              "   Delta: ", avg_delta,
              "   Error: ", mean_error)        
env.close()


