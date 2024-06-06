import gym
import numpy as np
import gym_donkeycar
from hp import *
from src.utils.config_loader import load_config, CONFIG_PATH

#* Initialize environment
conf = load_config(CONFIG_PATH)
env = gym.make("donkey-generated-track-v0", conf=conf)
obs, reward, done, info = env.reset()

for i in range(100):
        action = np.array([0.1, 0.1])
        obs, reward, done, info = env.step(action)
        print("--------------------------------")
        print("STEP: ", i)
        print(info["lidar"])
        print(type(info["lidar"]))
        print(info["lidar"].shape)
        print("--------------------------------")

env.close()

