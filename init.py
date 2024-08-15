from hyperparams.hp_config import *
import importlib

hp_config = load_hp_config()

#* Env wrapper import
ev_wrapper_name = hp_config["env_wrapper"]["name"]
env_wrapper_module = "src.environment.wrapper"

env_wrapper_module = importlib.import_module(env_wrapper_module)
Wrapper = getattr(env_wrapper_module, ev_wrapper_name)

#* Action wrapper import
action_wrapper_name = hp_config["action_wrapper"]["name"]
action_wrapper_module = "src.environment.action_shaping"
action_wrapper_params = hp_config["action_wrapper"]["parameters"]
action_wrapper_module = importlib.import_module(action_wrapper_module)
action_wrapper = getattr(action_wrapper_module, action_wrapper_name)

#* Agent import
agent_name = hp_config["Agent"]["name"]
agent_module = "src.agents"

agent_module = importlib.import_module(agent_module)
Agent = getattr(agent_module, agent_name)

import gym
import numpy as np
import wandb
import gym_donkeycar
from src.utils.config_loader import load_config, CONFIG_PATH
from src.utils.performance import PerformanceMSE as Performance

#* Initialize environment
conf = load_config(CONFIG_PATH)
env = gym.make("donkey-generated-track-v0", conf=conf)
obs, reward, done, info = env.reset()
start_action = np.array([0.0, 0.0])

#* Initialize action wrapper
action_wrapper = action_wrapper(**action_wrapper_params)

#* Initialize Env Wrapper
wrapper = Wrapper(state=obs,
    action = np.array([0.0, 0.0]),
    done = done,
    info = info,
    max_cte = conf["max_cte"],
    sigma = hp_config["env_wrapper"]["SIGMA"],
    action_cost = hp_config["env_wrapper"]["ACTION_COST"],
    target_speed = hp_config["env_wrapper"]["TARGET_SPEED"])

#* Initialize performanse
performance = Performance(ref_speed=hp_config["env_wrapper"]["TARGET_SPEED"])

#* Reset the wrapper
obs, reward, done = wrapper.reset(obs, start_action, 
                           done, info)

#* Initialize agent
