import os
import yaml

def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'model_configuration.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


MODEL_CONFIG = load_config()