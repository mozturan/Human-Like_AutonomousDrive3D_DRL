import gym
import numpy as np
import wandb
import gym_donkeycar

from src.environment.wrapper import Gnod as Wrapper
from src.utils.config_loader import load_config, CONFIG_PATH
from src.agents import sac
from hp import *

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

#* Initialize agent
agent = sac.SAC(state_size=obs.shape, 
                action_size=ACTION_SIZE, 
                hidden_size=HIDDEN_SIZE,
                min_size=MIN_SIZE, 
                batch_size=BATCH_SIZE,
                model_name=MODEL_NAME,
                max_action=MAX_ACTION, 
                temperature=TEMPERATURE)

#* Initialize wandb
wandb.init(
    # set the wandb project where this run will be logged

    project="Batch Size",
    name = "Noise_0.4",

    config={
            "architecture": "AE-MLP",
            "dataset": "Generated-Track-v0",
            "epochs": 100,
            "batch_size": BATCH_SIZE,
            "alpha": ALPHA,
            "beta": BETA,
            "network": NETWORK,
            "min mem size": MIN_MEM_SIZE,
            "max action": MAX_ACTION,
            "temperature": TEMPERATURE,
            "tau": TAU,
            "gamma": GAMMA,
            "reward scale": REWARD_SCALE,
            "minibatch size": BATCH_SIZE,
            "max_cte": conf["max_cte"],
            "sigma": SIGMA,
            "action_cost": ACTION_COST,
            "target_speed": TARGET_SPEED,
            "noise": NOISE,
            "env": ENV,
            "wrapper": wrapper}
)

#* Initialize variables
evaluate = False
score_history = []

#* Start training
for episode in range(5000):

        #* Reset environment and the wrapper
        obs, reward, done, info = env.reset()
        obs, reward, done = wrapper.reset(obs, np.array([0.0, 0.0]), 
                     done, info)
        
        #* Reset episode reward
        episode_reward = 0
        episode_len = 0

        #* Evaluate every 50 episodes
        if ((episode % 50 == 0) and (episode != 0)):
                evaluate = True

        #* Start episode
        while not done and episode_len < 400:

                #* Get action from agent and normalize it
                action = agent.choose_action(obs, evaluate = evaluate)
                normalized_action = [action[0], (action[1] / 2.0)+0.1]

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

        #* Update score history
        score_history.append(episode_reward)
        avg_score = np.mean(score_history)
        
        #* Log to wandb
        wandb.log({"episode_length": episode_len, 
                   "episode_reward": episode_reward, 
                   "score_avg": avg_score})
        
        #* Print Evaluation Results
        if evaluate or episode ==0:
                print("Evaluation")
                print("Episode {} Reward {} Episode Length {}".format(episode, episode_reward, episode_len))
                evaluate = False

                #* save model
                # agent.save(episode, wrapper)
    
env.close()