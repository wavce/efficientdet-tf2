import tensorflow as tf


def posterize(image, bits):
    shift = 8 - bits

    return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)