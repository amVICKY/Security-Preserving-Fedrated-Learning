import yaml
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

configuration_path = os.path.join(
    BASE_DIR,
    "configs",
    "config.yaml"
)

def load_config(config_path=configuration_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config

training_dict = load_config()