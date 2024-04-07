import logging
import os
import json

CONFIG_PATH = "src/utils/config.json"
#Open config.json and load all the parameters, than close the congif.json file
def load_config(config_path: str) -> dict:
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
