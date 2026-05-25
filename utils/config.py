import yaml

configuration_path = r"D:\INTERNSHIP_PROJECT\secure_federated_learning\configs\config.yaml"

def load_config(config_path=configuration_path):
    with open(config_path,"r") as file:
        config = yaml.safe_load(file)
    
    return config

traing_dict = load_config()
