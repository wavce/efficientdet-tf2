import tensorflow as tf


def regularization_loss(weights, weight_decay):
    weights = [tf.nn.l2_loss(w) for w in weights if "kernel" in w.name]

    loss = tf.add_n(weights) * weight_decay

    return loss


