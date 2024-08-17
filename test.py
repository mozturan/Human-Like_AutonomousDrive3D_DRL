from init import *
import keras
import tensorflow as tf

hp_config_path = "models/v43/Noisy-WeightedMovingAverage/hp_config.json"
model_string= "models/v43/Noisy-WeightedMovingAverage/_247.keras"

hp_config = config_init(hp_config_path)
Model = hp_config["Model"]
print(Model)
env, wrapper, action_wrapper = test_init(hp_config)

#* Load model
model = keras.saving.load_model(model_string, 
                                custom_objects=None, 
                                compile=True, 
                                safe_mode=True)

def chose_action (model, state):
        state = tf.convert_to_tensor([state])
        actions, _ = model.sample_normal(state, reparameterize=False)

        return actions[0]

max_episode_length = 1000
#* Start training
for episode in range(1000):

        action_wrapper.reset()
        #* Reset environment and the wrapper
        obs, reward, done, info = env.reset()
        obs, reward, done = wrapper.reset(obs, np.array([0.0, 0.0]), 
                     done, info)
        
        #* Reset episode reward
        episode_reward = 0
        episode_len = 0

        #* Start episode
        while not done and episode_len < max_episode_length:

                #* Get action from agent and normalize it
                action = chose_action(model, obs)
                normalized_action = action_wrapper.step(action)

                #* Step through environment and process the step
                new_obs, reward, done, new_info = env.step(np.array(normalized_action))
                new_obs, reward, done = wrapper.step(new_obs, action, 
                        done, new_info)
                                
                #* Update observation
                obs = new_obs
    
env.close()
