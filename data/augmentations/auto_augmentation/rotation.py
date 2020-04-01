import math
import numpy as np
import tensorflow as tf
from skimage import transform
from data.auto_augmentation import clip_box
from data.auto_augmentation import check_box_area
from data.auto_augmentation import scale_box_only_op_probability
from data.auto_augmentation import apply_multi_box_augmentation_wrapper


@tf.function
def _rotate(image, angle, replace, order=0):
    def _rot(img, agl, cval):
        h, w, _ = img.shape
        x_offset = ((w - 1) - (np.cos(agl) * (w - 1) - np.sin(agl) * (h - 1))) / 2.0
        y_offset = ((h - 1) - (np.sin(agl) * (w - 1) + np.cos(agl) * (h - 1))) / 2.0
        tfm = transform.ProjectiveTransform(matrix=np.array([[np.cos(agl), -np.sin(agl), x_offset],
                                                            [np.sin(agl), np.cos(agl), y_offset],
                                                            [0, 0, 1]]))
        img = transform.warp(img, tfm.inverse, order=order, cval=cval, preserve_range=True)
        # img = transform.rotate(img, agl, cval=cval, preserve_range=True)
        img = img.astype(np.uint8)

        return img

    replace = tf.convert_to_tensor(replace)
    angle = tf.convert_to_tensor(angle)

    return tf.numpy_function(_rot, [image, angle, replace], tf.uint8)


def rotate(image, angle, replace):
    """Rotates the image by angle either clockwise or counterclockwise.

    Args:
        image: An image Tensor of type uint8.
        angle: Float, a scalar angle in degrees to rotate all images by. If
            angle is positive the image will be rotated clockwise otherwise it will
            be rotated counterclockwise.
        replace: A one or three value 1D tensor to fill empty pixels caused by
            the rotate operation.
    Returns:
        The rotated version of image.
    """
    radians = angle * (math.pi / 180.)

    # In practice, we should randomize the rotation angles by flipping
    # it negatively half the time, but that"s done on `angle` outside
    # of the function.
    return _rotate(image, radians, replace)


def _rotate_box(box, image_shape, angle):
    """Rotates the box coordinates by angle.

    Args:
        box: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
            of type float that represents the normalized coordinates between
            0 and 1.
        image_shape: (height, width) of the image.
        angle: Float, a scalar angle to rotate all images by. If angle is
            positive, the image will rotated clockwise otherwise it will be
            rotated counterclockwise.

    Returns:
        A tensor of the same shape as box, but now with the rotated coordinates.
    """
    image_height = tf.cast(image_shape[0], tf.float32)
    image_width = tf.cast(image_shape[1], tf.float32)

    # Convert from angle to radians
    radians = -angle * (math.pi / 180.)
    # Translate the box to the center of the image and turn the normalized 0-1
    # coordinates to absolute pixel locations.
    # Y coordinates are made negative as the y axis of images goes down with
    # increasing pixel values, so we negate to make sure x axis and y axis points
    # are in the traditionally positive direction.
    min_y = -tf.cast(image_height * (box[0] - 0.5), tf.int32)
    min_x = tf.cast(image_width * (box[1] - 0.5), tf.int32)
    max_y = -tf.cast(image_height * (box[2] - 0.5), tf.int32)
    max_x = tf.cast(image_width * (box[3] - 0.5), tf.int32)

    coordinates = tf.stack(
        [[min_y, min_x], [min_y, max_x], [max_y, min_x], [max_y, max_x]])
    coordinates = tf.cast(coordinates, tf.float32)
    # Rotate the coordinates according to the rotation matrix clockwise if
    # radians is positive, else negative.
    rotation_matrix = tf.stack([[tf.math.cos(radians), tf.math.sin(radians)],
                                [-tf.math.sin(radians), tf.math.cos(radians)]])
    new_coords = tf.cast(tf.matmul(rotation_matrix, tf.transpose(coordinates)), tf.int32)
    # Find min/max values and convert them back to normalized 0-1 floats.
    min_y = -(tf.cast(tf.reduce_max(new_coords[0, :]), tf.float32) / image_height - 0.5)
    min_x = tf.cast(tf.reduce_min(new_coords[1, :]), tf.float32) / image_width + 0.5
    max_y = -(tf.cast(tf.reduce_min(new_coords[0, :]), tf.float32) / image_height - 0.5)
    max_x = tf.cast(tf.reduce_max(new_coords[1, :]), tf.float32) / image_width + 0.5

    # Clip the boxes to be sure the fall between [0, 1].
    min_y, min_x, max_y, max_x = clip_box(min_y, min_x, max_y, max_x)
    min_y, min_x, max_y, max_x = check_box_area(min_y, min_x, max_y, max_x)

    return tf.stack([min_y, min_x, max_y, max_x])


def rotate_with_boxes(image, boxes, angle, replace):
    """Equivalent of PIL Rotate that rotates the image and box.

    Args:
        image: 3D uint8 Tensor.
        boxes: 2D Tensor that is a list of the boxes in the image. Each box
            has 4 elements (min_y, min_x, max_y, max_x) of type float.
        angle: Float, a scalar angle to rotate all images by. If
            angle is positive the image will be rotated clockwise otherwise it will
            be rotated counterclockwise.
        replace: A one or three value 1D tensor to fill empty pixels.
    Returns:
        A tuple containing a 3D uint8 Tensor that will be the result of rotating
        image by angle. The second element of the tuple is boxes, where now
        the coordinates will be shifted to reflect the rotated image.
    """
    # Rotate the image.
    image = rotate(image, angle, replace)
    image = tf.cast(image, tf.uint8)

    # Convert box coordinates to pixel values
    image_shape = tf.shape(image)[0:2]
    boxes = tf.map_fn(lambda box: _rotate_box(box, image_shape, angle), boxes)

    return image, boxes


def rotate_only_boxes(image, boxes, prob, angle, replace):
    """Apply rotate to each box in the image with probability prob."""
    func_changes_box = False
    prob = scale_box_only_op_probability(prob)

    return apply_multi_box_augmentation_wrapper(
        image, boxes, prob, rotate, func_changes_box, angle, replace)

