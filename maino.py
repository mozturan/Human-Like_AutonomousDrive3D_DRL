import gym
import numpy as np
import wandb
import gym_donkeycar

from src.environment.wrapper import Nothing as Wrapper
from src.environment.action_shaping import LowPassFilter as action_wrapping
from src.utils.config_loader import load_config, CONFIG_PATH
from src.utils.performance import PerformanceMSE as Performance
from src.agents import sac
from hp import *

#* Initialize environment
conf = load_config(CONFIG_PATH)
env = gym.make("donkey-generated-track-v0", conf=conf)
obs, reward, done, info = env.reset()
start_action = np.array([0.0, 0.0])
action_wrapper = action_wrapping(smoothing_coef = 0.5)

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

#* Initialize agent
agent = sac.SAC(state_size=obs.shape, 
                action_size=ACTION_SIZE, 
                hidden_size=HIDDEN_SIZE,
                min_size=MIN_SIZE, 
                batch_size=BATCH_SIZE,
                max_action=MAX_ACTION, 
                temperature=TEMPERATURE,
                use_noise=USE_NOISE,
                tau=TAU,
                gamma=GAMMA,
                alpha=ALPHA,
                beta=BETA)

#* Initialize wandb
wandb.init(
    # set the wandb project where this run will be logged

    project="Generation 0",
    name = "NothingX",

    config={
            "architecture": "AE-MLP",
            "dataset": "Generated-Track-v0",
            "epochs": 100,
            "batch_size": BATCH_SIZE,
            "alpha": ALPHA,
            "beta": BETA,
            "network": NETWORK,
            "min mem size": MIN_SIZE,
            "max action": MAX_ACTION,
            "temperature": TEMPERATURE,
            "tau": TAU,
            "gamma": GAMMA,
            "hidden size": HIDDEN_SIZE,
            "reward scale": REWARD_SCALE,
            "minibatch size": BATCH_SIZE,
            "max_cte": conf["max_cte"],
            "sigma": SIGMA,
            "action_cost": ACTION_COST,
            "target_speed": TARGET_SPEED,
            "use noise": USE_NOISE,
            "use smoothing": USE_SMOOTHING,
            "env": ENV,
            "wrapper": wrapper}
)

#* Initialize variables
evaluate = False
score_history = []
max_episode_length = 500
best_score = -1000

#* Start training
for episode in range(701):

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
                
        agent.save(episode, "NothingX")
    
env.close()