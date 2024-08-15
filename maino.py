from init import *

hp_config_path = None
hp_config = config_init(hp_config_path)
Model = hp_config["Model"]
save_path = f"models/v43/{Model}/"

if hp_config_path is None:
        env, wrapper, action_wrapper, agent, performance = train_init(hp_config)
else:
        #! This project gets bigger each day, help me
        env, wrapper, action_wrapper = test_init()

agent.save(-1, save_path)
save_hp_config(hp_config, f"{save_path}/hp_config.json")

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

        #* Reset performance
        performance.reset()

        #* Start episode
        while not done and episode_len < max_episode_length:

                #* Get action from agent and normalize it
                action = agent.choose_action(obs, evaluate = evaluate)
                normalized_action = action_wrapper.step(action)

                #* Step through environment and process the step
                new_obs, reward, done, new_info = env.step(np.array(normalized_action))
                new_obs, reward, done = wrapper.step(new_obs, action, 
                        done, new_info)
                
                #* Update episode reward
                episode_reward += reward
                episode_len +=1

                #* Store step in replay memory
                agent.remember(obs, action, reward, 
                        new_obs, done)
                
                #* Train agent
                if not evaluate:
                        agent.train()

                #* Update observation
                obs = new_obs

                #* Update performance
                performance(cte = new_info["cte"], speed = new_info["speed"], action = normalized_action)

        #* Update score history
        score_history.append(episode_reward)
        avg_score = np.mean(score_history)
        cumilative_reward = np.mean(score_history[-100:])

        #* Log to wandb
        mean_error, cte_avg, speed_avg, avg_delta = performance.get_metrics()

        wandb.log({"episode_length": episode_len, 
                   "episode_reward": episode_reward, 
                   "score_avg": avg_score,
                   "cumilative_avg": cumilative_reward,
                   "mean_error": mean_error,
                   "cte_avg": cte_avg,
                   "speed_avg": speed_avg,
                   "avg_delta": avg_delta
                   })
        
        #* Save model
        if cumilative_reward > best_score:
                best_score = cumilative_reward
                print("Best Score: ", best_score, "   Episode: ", episode)
                
        agent.save(episode, save_path)
    
env.close()