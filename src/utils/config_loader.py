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
    with open(config_path) as config_file:
        config = json.load(config_file)
        print(config)
    return config
