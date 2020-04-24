from .base_config import Config
from configs.efficientdet_config import get_efficientdet_config


def build_configs(name):
    if name.startswith("efficientdet"):
        return get_efficientdet_config(name)

