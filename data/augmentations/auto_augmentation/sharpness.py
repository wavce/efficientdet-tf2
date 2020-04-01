import tensorflow as tf
from data.auto_augmentation import blend


def sharpness(image, factor):
    """Implements Sharpness function from PIL using TF ops."""
    orig_image = image

    image = tf.cast(image, tf.float32)

    # Make image 4D for conv operation
    image = tf.expand_dims(image, 0)
    # SMOOTH PIL kernel.
    kernel = tf.constant(
        [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32, shape=[3, 3, 1, 1]) / 13.
    # Tile across channel dimensions
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    strides = [1, 1, 1, 1]
    degenerate = tf.nn.depthwise_conv2d(image, kernel, strides, padding="VALID", dilations=[1, 1])
    degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
    degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), 0)

    # For the borders of the resulting image, fill in the values of original image.
    mask = tf.ones_like(degenerate)
    padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
    padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])

    result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

    return blend(result, orig_image, factor)

