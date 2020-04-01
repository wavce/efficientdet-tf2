import tensorflow as tf
from data.auto_augmentation import blend


def brightness(image, factor):
    degenerate = tf.zeros_like(image)

    return blend(degenerate, image, factor)
