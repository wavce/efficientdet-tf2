import tensorflow as tf
from data.auto_augmentation import scale_box_only_op_probability
from data.auto_augmentation import apply_multi_box_augmentation_wrapper


def equalize(image):
    """Implements Equalize function from PIL using TF ops."""
    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = tf.cast(im[:, :, c], tf.int32)
        # Compute the histogram of the image channel.
        hist = tf.histogram_fixed_width(im, [0, 255], nbins=256)

        # For the purposes of computing the step, filter out the non-zeros.
        nonzero = tf.where(tf.not_equal(hist, 0))
        nonzero_hist = tf.reshape(tf.gather(hist, nonzero), [-1])
        step = (tf.reduce_sum(nonzero_hist) - nonzero_hist[-1]) // 255

        def build_lut(hist, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (tf.cumsum(hist) + (step // 2)) // step
            # Shift lut, prepending with 0
            lut = tf.concat([[0], lut[:-1]], 0)
            # Clip the counts to be in range. This is done in the C code for image.point
            return tf.clip_by_value(lut, 0, 255)

        # If step is zero, return the original image. Otherwise, build
        # lut from the full histogram and step and then index from it.
        result = tf.cond(tf.equal(step, 0),
                         lambda: im,
                         lambda: tf.gather(build_lut(hist, step), im))

        return tf.cast(result, tf.uint8)

    # Assumes RGB for now. Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = tf.stack([s1, s2, s3], 2)

    return image


def equalize_only_boxes(image, boxes, prob):
    """Apply equalize to each box in the image with probability prob."""
    func_changes_box = False
    prob = scale_box_only_op_probability(prob)

    return apply_multi_box_augmentation_wrapper(
        image, boxes, prob, equalize, func_changes_box)
