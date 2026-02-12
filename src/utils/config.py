import yaml


def load_config(path="./config/config.yaml"):
    with open(path) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as e:
            raise RuntimeError(f"Failed to load YAML file: {e}")
