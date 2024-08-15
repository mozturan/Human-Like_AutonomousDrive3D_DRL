import json
import logging
import os

def load_hp_config(config_path: str = "/home/o/Documents/donkeycar_rl/hyperparams/hp_config.json") -> dict:
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

def save_hp_config(config: dict, config_path: str):
    """
    Save a configuration to a JSON file.

    Args:
        config (dict): The configuration to be saved.
        config_path (str): The path to the configuration file.
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=4)
