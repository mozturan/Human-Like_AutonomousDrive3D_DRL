from utils import load_sim_config, \
    load_train_config, save_test_config, \
        Performance
import gym, gym_donkeycar
import numpy as np
import importlib
import keras
import tensorflow as tf

max_episode_length = 500
test_config_path = "models/final_destination/SAC-CL+Noise/test_config.json"
model_path = "models/final_destination/SAC-CL+Noise/_361.keras"

test_config = load_train_config(test_config_path)

conf = load_sim_config()
env = gym.make(test_config["Env"]["name"], conf=conf)
obs,reward, done, info = env.reset()

wrappers_module = "wrappers"
state_wrapper = test_config["state_wrapper"]["name"]
action_wrapper = test_config["action_wrapper"]["name"]
reward_wrapper = test_config["reward_wrapper"]["name"]
wrappers_module = importlib.import_module(wrappers_module)
state_wrapper = getattr(wrappers_module, state_wrapper)
action_wrapper = getattr(wrappers_module, action_wrapper)
reward_wrapper = getattr(wrappers_module, reward_wrapper)

#* Initialize wrappers
state_wrapper = state_wrapper(**test_config["state_wrapper"]["parameters"])
action_wrapper = action_wrapper(**test_config["action_wrapper"]["parameters"])
reward_wrapper = reward_wrapper(**test_config["reward_wrapper"]["parameters"])

obs = state_wrapper.reset(obs, 0.0, done, info)
action_wrapper.reset()
reward_wrapper.reset()

#* Initialize performance
performance = Performance(ref_cte=conf["max_cte"],
                          ref_speed=test_config["reward_wrapper"]["parameters"]["target_speed"])

#* Load model
model = keras.saving.load_model(model_path, 
                                custom_objects=None, 
                                compile=True, 
                                safe_mode=True)

def chose_action (model, state):
        state = tf.convert_to_tensor([state])
        actions, _ = model.sample_normal(state, reparameterize=False)

        return actions[0]

for episode in range(1):

        #* Reset environment and the wrapper
        obs, reward, done, info = env.reset()
        obs = state_wrapper.reset(obs, 0.0, done, info)
        action_wrapper.reset()
        reward_wrapper.reset()

        episode_len = 0

        while not done and episode_len < max_episode_length:

                #* Get action from agent and normalize it
                action = chose_action(model, obs)
                filtered_action = action_wrapper(action)

                #* Step through environment and process the step
                new_obs, reward, done, new_info \
                    = env.step(np.array(filtered_action))

                new_obs = state_wrapper(new_obs, action, done, new_info)
                reward, done = reward_wrapper(action, new_info, done)

                #* Update performance
                performance(cte = new_info["cte"], speed = new_info["speed"], action = filtered_action)
                episode_len +=1

                #* Update observation
                obs = new_obs

        print(test_config_path)
        mean_error, cte_avg, speed_avg, avg_delta = performance.get_metrics()
        print("Mean Error: ", mean_error,
              "   CTE: ", cte_avg,
              "   Speed: ", speed_avg,
              "   Delta: ", avg_delta)
        
        print("Episode Len:", episode_len)

env.close()



