import tensorflow as tf
from data.auto_augmentation import blend


def contrast(image, factor):
    degenerate = tf.image.rgb_to_grayscale(image)
    # Cast before calling tf.histogram
    degenerate = tf.cast(degenerate, tf.int32)

    # Compute the grayscale histogram, the compute the mean pixel value,
    # and create a constant image size of that value. Use that as the blending
    # degenerate target of the original image.
    hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
    mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0

    degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
    degenerate = tf.clip_by_value(degenerate, 0, 255)
    degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))

    return blend(degenerate, image, factor)


def autocontrast(image):
    """Implements Autcontrast function from PIL using TF ops.

    Args:
        image: A 3D uint8 tensor.

    Returns:
        The image after it has had autocontrast applied to it
        and will be of type uint8.
    """
    def scale_channel(img):
        """Scale the 2D image using the autocontrast rule."""
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        lo = tf.cast(tf.reduce_min(image), tf.float32)
        hi = tf.cast(tf.reduce_max(image), tf.float32)

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(im):
            scale = 255. / (hi - lo)
            offset = -lo * scale
            im = tf.cast(im, tf.float32) * scale + offset
            im = tf.clip_by_value(im, 0., 255.)

            return tf.cast(im, tf.uint8)

        result = tf.cond(hi > lo, lambda: scale_values(img), lambda: image)

        return result

    # Assumes RGB for now, Scales each channel independently
    # and the stack the result
    s1 = scale_channel(image[:, :, 0])
    s2 = scale_channel(image[:, :, 1])
    s3 = scale_channel(image[:, :, 2])

    image = tf.stack([s1, s2, s3], 2)

    return image
