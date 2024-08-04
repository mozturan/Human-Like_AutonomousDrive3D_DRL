import gym
import numpy as np
import gym_donkeycar
import wandb

from hp import *
import keras
import tensorflow as tf

from src.utils.config_loader import load_config, CONFIG_PATH
from src.environment.wrapper import Faith as Wrapper
from src.environment.action_shaping import SmoothingAction
from src.utils.performance import PerformanceMSE as Performance

model_string= "models/Horace/actor_[450]_Horace.keras"

#* Load model
model = keras.saving.load_model(model_string, 
                                custom_objects=None, 
                                compile=True, 
                                safe_mode=True)

def chose_action (model, state):
        state = tf.convert_to_tensor([state])
        actions, _ = model.sample_normal(state, reparameterize=False)

        return actions[0]

#* Initialize environment
conf = load_config(CONFIG_PATH)
env = gym.make("donkey-generated-track-v0", conf=conf)
obs, reward, done, info = env.reset()
start_action = np.array([0.0, 0.0])
action_wrapper = SmoothingAction(smoothing_coef = 0.5)

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

#* Initialize wandb
wandb.init(
    # set the wandb project where this run will be logged

    project="Alpha Tests",
    name = "VanillaN-F-S",

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
            "hidden size": HIDDEN_SIZE,
            "reward scale": REWARD_SCALE,
            "minibatch size": BATCH_SIZE,
            "max_cte": conf["max_cte"],
            "sigma": SIGMA,
            "action_cost": ACTION_COST,
            "target_speed": TARGET_SPEED,
            "use noise": USE_NOISE,
            "use smoothing": USE_SMOOTHING,
            "noise": NOISE,
            "env": ENV,
            "wrapper": wrapper}
)
