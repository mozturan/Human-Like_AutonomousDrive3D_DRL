from utils import load_sim_config, load_train_config, save_test_config, Performance
import gym, gym_donkeycar
import numpy as np
import wandb, importlib

score_history = []
max_episode_length = 500
train_config = load_train_config()

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
#! RESET WRAPPERS

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
performance = Performance()

