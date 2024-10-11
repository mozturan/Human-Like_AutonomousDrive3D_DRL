from utils import load_sim_config, load_train_config, save_test_config, Performance
import gym, gym_donkeycar
import numpy as np
import wandb, importlib

score_history = []
max_episode_length = 500
train_config = load_train_config()

env_wrapper_name = train_config["env_wrapper"]["name"]
env_wrapper_module = "wrappers"
env_wrapper_module = importlib.import_module(env_wrapper_module)
wrapper = getattr(env_wrapper_module, env_wrapper_name)

env = gym.make("donkey-generated-track-v0", conf=load_sim_config())
env = wrapper(env)