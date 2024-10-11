from init import *
import keras
import tensorflow as tf

hp_config_path = "models/v43/Noisy-WeightedMovingAverage/hp_config.json"
model_string= "models/v43/Noisy-WeightedMovingAverage/_215.keras"

hp_config = config_init(hp_config_path)
Model = hp_config["Model"]
print(Model)
env, wrapper, action_wrapper, performance = test_init(hp_config)

#* Load model
model = keras.saving.load_model(model_string, 
                                custom_objects=None, 
                                compile=True, 
                                safe_mode=True)

def chose_action (model, state):
        state = tf.convert_to_tensor([state])
        actions, _ = model.sample_normal(state, reparameterize=False)

        return actions[0]

max_episode_length = 10000
#* Start training
for episode in range(1):

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

                #* Update performance
                performance(cte = new_info["cte"], speed = new_info["speed"], action = normalized_action)
                episode_len +=1

                if new_info["last_lap_time"] > 0.0:
                        print("last lap time: ", new_info["last_lap_time"])
                        break


        print("parametreler")
        mean_error, cte_avg, speed_avg, avg_delta = performance.get_metrics()
        print("Mean Error: ", mean_error,
              "   CTE: ", cte_avg,
              "   Speed: ", speed_avg,
              "   Delta: ", avg_delta)
        
        print("Episode Len:", episode_len)

env.close()
