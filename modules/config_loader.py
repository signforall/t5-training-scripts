import yaml
import argparse

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))  # Recursively convert nested dictionaries
            else:
                setattr(self, key, value)

def load_config():
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    with open(args.config_path, "r") as file:
        config_dict = yaml.safe_load(file)

    return Config(config_dict)  # Return an object instead of a dict
