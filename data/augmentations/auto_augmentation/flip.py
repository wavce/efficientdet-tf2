import tensorflow as tf
from data.auto_augmentation.common import scale_box_only_op_probability
from data.auto_augmentation.common import apply_multi_box_augmentation_wrapper


def flip_only_boxes(image, boxes, prob):
    """Apply flip_lr to each box in the image with probability prob."""
    func_changes_box = False
    prob = scale_box_only_op_probability(prob)

    return apply_multi_box_augmentation_wrapper(
        image, boxes, prob, tf.image.flip_left_right, func_changes_box)
