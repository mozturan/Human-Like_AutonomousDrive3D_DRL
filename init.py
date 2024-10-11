from hyperparams.hp_config import *
import importlib
import gym
import numpy as np
import wandb
import gym_donkeycar
from src.utils.config_loader import load_config, CONFIG_PATH
from src.utils.performance import PerformanceMSE as Performance
 
#* Initialize variables
evaluate = False
score_history = []
max_episode_length = 500
best_score = -1000

def config_init(hp_config_path = None):
    if hp_config_path == None:
        hp_config = load_hp_config()
    else:
        hp_config = load_hp_config(hp_config_path)
    return hp_config

def train_init(hp_config):

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
    agent_module = f"src.agents.{agent_name}"

    Agent = importlib.import_module(agent_module)

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
    agent = Agent.SAC(state_size=obs.shape, 
                    action_size=hp_config["Agent"]["ACTION_SIZE"],
                    hidden_size=hp_config["Agent"]["HIDDEN_SIZE"],
                    min_size=hp_config["Agent"]["MIN_SIZE"],
                    batch_size=hp_config["Agent"]["BATCH_SIZE"],
                    max_action=hp_config["Agent"]["MAX_ACTION"],
                    temperature=hp_config["Agent"]["TEMPERATURE"],
                    tau=hp_config["Agent"]["TAU"],
                    gamma=hp_config["Agent"]["GAMMA"],
                    use_noise=hp_config["Agent"]["USE_NOISE"],
                    # alpha = hp_config["Agent"]["ALPHA"],
                    # beta = hp_config["Agent"]["BETA"])
                    )

    #* Initialize wandb
    wandb.init(
        project="43.0.0",
        name = hp_config["Model"],

        config= hp_config)
    
    return env, wrapper, action_wrapper, agent, performance


def test_init(hp_config):
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


    return env, wrapper, action_wrapper, performance