import tensorflow as tf
from data.auto_augmentation import scale_box_only_op_probability
from data.auto_augmentation import apply_multi_box_augmentation_wrapper


def solarize(image, threshold=128):
    """For each pixel in the image, select the pixel if the value is less
        than the threshold. Otherwise, subtract 255 from the pixel.
    """
    return tf.where(image < threshold, image, 255 - image)


def solarize_add(image, addition=0, threshold=128):
    """For each pixel in the image less than threshold, we add `addition`
    amount to it and the clip the pixel value to the between 0 and 255.
    The value of `addition` is between -128 and 128.
    """
    added_image = tf.cast(image, tf.int64) + addition
    added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), tf.uint8)

    return tf.where(image < threshold, added_image, image)


def solarize_only_boxes(image, boxes, prob, threshold):
    """Apply solarize to each box in the image with probability prob."""
    func_changes_box = False
    prob = scale_box_only_op_probability(prob)

    return apply_multi_box_augmentation_wrapper(
        image, boxes, prob, solarize, func_changes_box, threshold)
