import tensorflow as tf


def build_learning_rate_scheduler(**kwargs):
    scheduler = kwargs.pop("learning_rate_scheduler")
    warmup_steps = kwargs.pop("warmup_steps")
    if scheduler == "piecewise_constant":
        boundaries = kwargs.pop("boundaries")
        values = kwargs.pop("values")
        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=boundaries, values=values)
    elif scheduler == "cosine":
        initial_learning_rate = kwargs.pop("initial_learning_rate")
        decay_steps = kwargs.pop("steps") - warmup_steps
        return tf.keras.experimental.CosineDecay(
            initial_learning_rate=initial_learning_rate, decay_steps=decay_steps)
    else:
        raise TypeError("Could not interpret learning rate scheduler"
                        " function identifier: {}".format(repr(scheduler)))
