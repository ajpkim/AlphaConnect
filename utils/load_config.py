import yaml

from utils.dot_dict import DotDict

def load_config(config_file):
    with open(config_file, mode='r') as f:
        config = yaml.safe_load(f)
        config = DotDict(config)
    return config
