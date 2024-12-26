from utils import load_sim_config, \
    load_train_config, save_test_config, \
        Performance
import gym, gym_donkeycar
import numpy as np
import wandb, importlib
import os

actor_loss_history = []
critic_loss_history = []
score_history = []
max_episode_length = 300
best_score = -1000
train_config = load_train_config("config/train_config_noise_.json")

# create folder if not exist
save_path = f"models/holy_nation/{train_config['Model']}"
if not os.path.exists(save_path):
    os.makedirs(save_path)

#* Initialize environment
conf = load_sim_config()
env = gym.make(train_config["Env"]["name"], conf=conf)
obs,reward, done, info = env.reset()

#* Import wrappers
wrappers_module = "wrappers"
state_wrapper = train_config["state_wrapper"]["name"]
action_wrapper = train_config["action_wrapper"]["name"]
reward_wrapper = train_config["reward_wrapper"]["name"]
wrappers_module = importlib.import_module(wrappers_module)
state_wrapper = getattr(wrappers_module, state_wrapper)
action_wrapper = getattr(wrappers_module, action_wrapper)
reward_wrapper = getattr(wrappers_module, reward_wrapper)

#* Initialize wrappers
state_wrapper = state_wrapper(**train_config["state_wrapper"]["parameters"])
action_wrapper = action_wrapper(**train_config["action_wrapper"]["parameters"])
reward_wrapper = reward_wrapper(**train_config["reward_wrapper"]["parameters"])

obs = state_wrapper.reset(obs, 0.0, done, info)
action_wrapper.reset()
reward_wrapper.reset()

#* Import agent
agent_name = train_config["Agent"]["name"]
agent_module = "agents"
agent_module = importlib.import_module(agent_module)
agent = getattr(agent_module, agent_name)

#* Initialize agent
agent = agent(state_size = obs.shape, 
              action_size = train_config["Agent"]["ACTION_SIZE"], 
              hidden_size = train_config["Agent"]["HIDDEN_SIZE"],
              min_size = train_config["Agent"]["MIN_SIZE"],
              batch_size = train_config["Agent"]["BATCH_SIZE"],
              max_action = train_config["Agent"]["MAX_ACTION"],
              temperature = train_config["Agent"]["TEMPERATURE"],
              tau = train_config["Agent"]["TAU"],
              gamma = train_config["Agent"]["GAMMA"],
              use_noise = train_config["Agent"]["USE_NOISE"])

#* Initialize performance
performance = Performance(ref_cte=conf["max_cte"],
                          ref_speed=train_config["reward_wrapper"]["parameters"]["target_speed"])

wandb.init(
    project = "holy_nation",
    name = train_config["Model"],
    config = train_config)

#* saving model config
agent.save(-1, save_path)
save_test_config(train_config, 
                 f"{save_path}/test_config.json")

for episode in range(250):
        
        last_action = np.zeros(train_config["Agent"]["ACTION_SIZE"])
        #* Reset environment and the wrapper
        obs, reward, done, info = env.reset()
        obs = state_wrapper.reset(obs, 0.0, done, info)
        action_wrapper.reset()
        reward_wrapper.reset()

        episode_reward = 0
        episode_len = 0

        while not done and episode_len < max_episode_length:

                #* Get action from agent and normalize it
                action = agent.choose_action(obs, evaluate = True)
                filtered_action = action_wrapper(action)

                #* Step through environment and process the step
                new_obs, reward, done, new_info \
                    = env.step(np.array(filtered_action))

                new_obs = state_wrapper(new_obs, action, done, new_info)
                reward, done = reward_wrapper(action, new_info, done)

                #* Update episode reward
                episode_reward += reward
                episode_len +=1

                #* Store step in replay memory
                agent.remember(obs, action, reward, 
                        new_obs, done, last_action)

                actor_loss, critic_loss = agent.train()
                actor_loss_history.append(actor_loss)
                critic_loss_history.append(critic_loss)

                obs = new_obs
                last_action = action

                performance(cte = new_info["cte"], 
                            speed = new_info["speed"], 
                            action = filtered_action)

        score_history.append(episode_reward)
        avg_score = np.mean(score_history)
        cumilative_last100_reward = np.mean(score_history[-100:])

        #* Log to wandb
        mean_error, cte_avg, speed_avg, avg_delta = performance.get_metrics()

        wandb.log({"episode_length": episode_len, 
                   "episode_reward": episode_reward, 
                   "score_avg": avg_score,
                   "cumilative_avg": cumilative_last100_reward,
                   "mean_error": mean_error,
                   "cte_avg": cte_avg,
                   "speed_avg": speed_avg,
                   "avg_delta": avg_delta
                   })
        
        if cumilative_last100_reward > best_score:
                best_score = cumilative_last100_reward
                print("Best Score: ", best_score, "   Episode: ", episode)

        agent.save(episode, save_path)

actor_loss_history = np.array(actor_loss_history)
critic_loss_history = np.array(critic_loss_history)
np.save(f"{save_path}/actor_loss.npy", actor_loss_history)
np.save(f"{save_path}/critic_loss.npy", critic_loss_history)

env.close()