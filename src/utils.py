import yaml


def read_config_params(config_path: str) -> dict:
    """
    Read config yaml file
    """
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    return config
