import tensorflow as tf
from .lookahead_optimizer import LookaheadOptimizer
from .accum_optimizer import AccumOptimizer


OPTIMIZERS = {
    "sgd": tf.keras.optimizers.SGD,
    "adam": tf.keras.optimizers.Adam,
    "rmsprop": tf.keras.optimizers.RMSprop
}


def build_optimizer(**kwargs):
    optimizer = kwargs.pop("optimizer")
    return OPTIMIZERS[optimizer](**kwargs)

