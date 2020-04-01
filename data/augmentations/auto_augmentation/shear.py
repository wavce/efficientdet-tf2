import numpy as np
import tensorflow as tf
from skimage import transform
from data.auto_augmentation import clip_box
from data.auto_augmentation import check_box_area
from data.auto_augmentation import scale_box_only_op_probability
from data.auto_augmentation import apply_multi_box_augmentation_wrapper


def shear_x(image, level, replace, order=0):
    """The ShearX operation shears the image along the horizontal axis with `level`
    magnitude.

    Args:
        image: image tensor.
        level: Strength of the operation specified as an Integer from
        [0, `PARAMETER_MAX`].
        replace: value for replacing empty
        order : int, optional
                The order of interpolation. The order has to be in the range 0-5:
                - 0: Nearest-neighbor
                - 1: Bi-linear (default)
                - 2: Bi-quadratic
                - 3: Bi-cubic
                - 4: Bi-quartic
                - 5: Bi-quintic

    Returns:
        A Image that has had ShearX applied to it.
    """
    level = tf.convert_to_tensor(level)
    replace = tf.convert_to_tensor(replace)

    def _shear(img, lvl, cval):
        # val = float_parameter(val, 0.3)
        tfm = transform.ProjectiveTransform(
            matrix=np.array([[1., lvl, 0.], [0., 1., 0.], [0, 0., 1.]]))

        img = transform.warp(img, tfm.inverse, order=order, cval=cval, preserve_range=True)
        img = img.astype(np.uint8)

        return img

    @tf.function
    def func(img):
        return tf.numpy_function(_shear, [img, level, replace], Tout=img.dtype)

    return func(image)


def shear_y(image, level, replace, order=0):
    """The ShearY operation shears the image along the vertical axis with `level`
    magnitude.

    Args:
        image: Image tensor.
        level: Strength of the operation specified as an Integer from
            [0, `PARAMETER_MAX`].
        replace: flat value for replacing empty.
        order : int, optional
                The order of interpolation. The order has to be in the range 0-5:
                - 0: Nearest-neighbor
                - 1: Bi-linear (default)
                - 2: Bi-quadratic
                - 3: Bi-cubic
                - 4: Bi-quartic
                - 5: Bi-quintic

    Returns:
        A Image that has had ShearX applied to it."""
    level = tf.convert_to_tensor(level)
    replace = tf.convert_to_tensor(replace)

    def _shear(img, lvl, cval):
        # val = float_parameter(val, 0.3)
        tfm = transform.ProjectiveTransform(
            matrix=np.array([[1., 0., 0.], [lvl, 1., 0.], [0, 0., 1.]]))

        img = transform.warp(img, tfm.inverse, order=order, cval=cval, preserve_range=True)
        img = img.astype(np.uint8)

        return img

    @tf.function
    def func(img):
        return tf.numpy_function(_shear, [img, level, replace], Tout=img.dtype)

    return func(image)


def _shear_box(box, image_shape, level, shear_horizontal):
    """Shifts the box according to how the image was sheared.

    Args:
        box: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
            of type float that represents the normalized coordinates between 0 and 1.
        image_shape: Int, (height, width) of the image.
        level: Float. How much to shear the image.
        shear_horizontal: If true then shear in X dimension else shear in
            the Y dimension.

    Returns:
        A tensor of the same shape as box, but now with the shifted coordinates.
    """
    image_height, image_width = tf.cast(image_shape[0], tf.float32), tf.cast(image_shape[1], tf.float32)

    # Change box coordinates to be pixels.
    min_y = tf.cast(image_height * box[0], tf.int32)
    min_x = tf.cast(image_width * box[1], tf.int32)
    max_y = tf.cast(image_height * box[2], tf.int32)
    max_x = tf.cast(image_width * box[3], tf.int32)
    coordinates = tf.stack([[min_y, min_x], [min_y, max_x], [max_y, min_x], [max_y, max_x]])
    coordinates = tf.cast(coordinates, tf.float32)

    # Shear the coordinates according to the translation matrix
    if shear_horizontal:
        translation_matrix = tf.stack([[1, 0], [level, 1]])
    else:
        translation_matrix = tf.stack([[1, level], [0, 1]])

    new_coords = tf.cast(tf.matmul(translation_matrix, tf.transpose(coordinates)), tf.int32)

    # Find min/max values and convert them back to floats
    min_y = tf.cast(tf.reduce_min(new_coords[0, :]), tf.float32) / image_height
    min_x = tf.cast(tf.reduce_min(new_coords[1, :]), tf.float32) / image_width
    max_y = tf.cast(tf.reduce_max(new_coords[0, :]), tf.float32) / image_height
    max_x = tf.cast(tf.reduce_max(new_coords[1, :]), tf.float32) / image_width

    # Clip the boxes to be sure the fall between [0, 1].
    min_y, min_x, max_y, max_x = clip_box(min_y, min_x, max_y, max_x)
    min_y, min_x, max_y, max_x = check_box_area(min_y, min_x, max_y, max_x)

    return tf.stack([min_y, min_x, max_y, max_x])


def shear_with_boxes(image, boxes, level, replace, shear_horizontal):
    """Applies Shear Transformation to the image and shifts the boxes.

    Args:
        image: 3D uint8 Tensor.
        boxes: 2D Tensor that is a list of the boxes in the image. Each box
            has 4 elements (min_y, min_x, max_y, max_x) of type float with values
            between [0, 1].
        level: Float. How much to shear the image. This value will be between
            -0.3 to 0.3.
        replace: A one or three value 1D tensor to fill empty pixels.
        shear_horizontal: Boolean. If true then shear in X dimension else shear in
        the Y dimension.

    Returns:
        A tuple containing a 3D uint8 Tensor that will be the result of shearing
        image by level. The second element of the tuple is boxes, where now
        the coordinates will be shifted to reflect the sheared image.
    """
    if shear_horizontal:
        image = shear_x(image, level, replace)
    else:
        image = shear_y(image, level, replace)

    # Convert box coordinates to pixel values.
    image_shape = tf.shape(image)[0:2]
    boxes = tf.map_fn(lambda box: _shear_box(box, image_shape, level, shear_horizontal), boxes)

    return image, boxes


def shear_x_only_boxes(image, boxes, prob, level, replace):
    """Apply shear_x to each box in the image with probability prob."""
    func_changes_box = False
    prob = scale_box_only_op_probability(prob)

    return apply_multi_box_augmentation_wrapper(
        image, boxes, prob, shear_x, func_changes_box, level, replace)


def shear_y_only_boxes(image, boxes, prob, level, replace):
    """Apply shear_y to each box in the image with probability prob."""
    func_changes_box = False
    prob = scale_box_only_op_probability(prob)

    return apply_multi_box_augmentation_wrapper(
        image, boxes, prob, shear_y, func_changes_box, level, replace)

