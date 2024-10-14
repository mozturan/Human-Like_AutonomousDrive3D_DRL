import logging
import os
import json

CONFIG_PATH = "config/sim_config.json"
#Open config.json and load all the parameters, than close the congif.json file
def load_sim_config(config_path: str = CONFIG_PATH) -> dict:
    """
    Load configuration from a JSON file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        dict: A dictionary representing the JSON data read from the configuration file.
    """
    if isinstance(config_path, str):
        if os.path.isfile(config_path):
            try:
                with open(config_path) as config_file:
                    config = json.load(config_file)
                    logging.info(config)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logging.error(f"Error loading configuration: {e}")
                config = {}
        else:
            raise FileNotFoundError("Invalid file path: {}".format(config_path))
    else:
        raise ValueError("config_path must be a string.")
    return config

def load_train_config(config_path: str = "/config/train_config.json") -> dict:
    """
    Load configuration from a JSON file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        dict: A dictionary representing the JSON data read from the configuration file.
    """
    if isinstance(config_path, str):
        if os.path.isfile(config_path):
            try:
                with open(config_path) as config_file:
                    config = json.load(config_file)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logging.error(f"Error loading configuration: {e}")
                config = {}
        else:
            raise FileNotFoundError("Invalid file path: {}".format(config_path))
    else:
        raise ValueError("config_path must be a string.")
    return config

def save_test_config(config: dict, config_path: str):
    """
    Save a configuration to a JSON file.

    Args:
        config (dict): The configuration to be saved.
        config_path (str): The path to the configuration file.
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=4)

