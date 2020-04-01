import tensorflow as tf
from data.auto_augmentation import blend


def color(image, factor):
    degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))

    return blend(degenerate, image, factor)
