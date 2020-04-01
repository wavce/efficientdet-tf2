from configs import efficientdet_config


CFG_DICT = {
    "efficientdet": efficientdet_config
}


def build_configs(name):
    return CFG_DICT[name].CFG

