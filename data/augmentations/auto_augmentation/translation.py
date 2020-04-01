import numpy as np
import tensorflow as tf
from skimage import transform
from data.auto_augmentation import wrap
from data.auto_augmentation import unwrap
from data.auto_augmentation import clip_box
from data.auto_augmentation import check_box_area
from data.auto_augmentation import scale_box_only_op_probability
from data.auto_augmentation import apply_multi_box_augmentation_wrapper


def translate(image, level, replace, order=0):
    level = tf.convert_to_tensor(level)
    replace = tf.convert_to_tensor(replace)

    def _translate(img, lvl, cval):
        # val = float_parameter(val, 0.3)
        tfm = transform.ProjectiveTransform(np.array([[1., 0., lvl[0]],
                                                      [0, 1., lvl[1]],
                                                      [0, 0., 1.]]))

        img = transform.warp(img, tfm.inverse, cval=cval, order=order, preserve_range=True)
        img = img.astype(np.uint8)

        return img

    @tf.function
    def func(img):
        return tf.numpy_function(_translate, [img, level, replace], Tout=img.dtype)

    return func(image)


def translate_x(image, level, replace, order=0):
    return translate(image, [-level, 0], replace, order)


def translate_y(image, level, replace, order=0):
    return translate(image, [0, -level], replace, order)


def _shift_box(box, image_shape, pixels, shift_horizontal):
    """Shifts the box coordinates by pixels.

    Args:
        box: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
            of type float that represents the normalized coordinates between
            0 and 1.
        image_shape: the (height, width) of image.
        pixels: An int, how many pixels to shift the box.
        shift_horizontal: Boolean, if true, the in X dimension else shift
            in Y dimension.

    Returns:
        A tensor of the same shape as box, but now with the shifted coordinates.
    """
    pixels = tf.cast(pixels, tf.int32)

    # Convert box to integer pixel locations.
    min_y = tf.cast(tf.cast(image_shape[0], tf.float32) * box[0], tf.int32)
    min_x = tf.cast(tf.cast(image_shape[1], tf.float32) * box[1], tf.int32)
    max_y = tf.cast(tf.cast(image_shape[0], tf.float32) * box[2], tf.int32)
    max_x = tf.cast(tf.cast(image_shape[1], tf.float32) * box[3], tf.int32)

    if shift_horizontal:
        min_x = tf.maximum(0, min_x - pixels)
        max_x = tf.minimum(image_shape[1], max_x - pixels)
    else:
        min_y = tf.maximum(0, min_y - pixels)
        max_y = tf.minimum(image_shape[0], max_y - pixels)

    # Convert box back to floats.
    min_y = tf.cast(min_y, tf.float32) / tf.cast(image_shape[0], tf.float32)
    min_x = tf.cast(min_x, tf.float32) / tf.cast(image_shape[1], tf.float32)
    max_y = tf.cast(max_y, tf.float32) / tf.cast(image_shape[0], tf.float32)
    max_x = tf.cast(max_x, tf.float32) / tf.cast(image_shape[1], tf.float32)

    # Clip the boxes to be sure the fall between [0, 1].
    min_y, min_x, max_y, max_x = clip_box(min_y, min_x, max_y, max_x)
    min_y, min_x, max_y, max_x = check_box_area(min_y, min_x, max_y, max_x)

    return tf.stack([min_y, min_x, max_y, max_x])


def translate_boxes(image, boxes, pixels, replace, shift_horizontal):
    """Equivalent of PIL Translate in X/Y dimension that shifts image and box.

    Args:
        image: 3D uint8 Tensor.
        boxes: 2D Tensor that is a list of the boxes in the image. Each box
            has 4 elements (min_y, min_x, max_y, max_x) of type float with values
            between [0, 1].
        pixels: An int. How many pixels to shift the image and boxes
        replace: A one or three value 1D tensor to fill empty pixels.
        shift_horizontal: Boolean. If true then shift in X dimension else shift in
            Y dimension.

    Returns:
        A tuple containing a 3D uint8 Tensor that will be the result of translating
        image by pixels. The second element of the tuple is boxes, where now
        the coordinates will be shifted to reflect the shifted image.
    """
    if shift_horizontal:
        image = translate_x(image, pixels, replace)
    else:
        image = translate_y(image, pixels, replace)

    # Convert box coordinates to pixel values.
    image_shape = tf.shape(image)[0:2]

    boxes = tf.map_fn(lambda box: _shift_box(box, image_shape, pixels, shift_horizontal),
                      boxes)

    return image, boxes


def random_shift_box(image, box, pixel_scaling, replace, new_min_box_coords=None):
    """Move the box and the image content to a slightly new random location.

    Args:
        image: 3D uint8 Tensor.
        box: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x) of type float
            that represents the normalized coordinates between 0 and 1. The potential
            values for the new min corner of the box will be between
            [old_min - pixel_scaling * box_height / 2, old_min + pixel_scaling * box_height / 2].
        pixel_scaling: A float between 0 and 1 that specifies the pixel range that
            the new box location will be sampled from.
        replace: A one or three value 1D tensor to fill empty pixels.
        new_min_box_coords: If not None, the this is a tuple that specifies the (min_y, min_x)
            coordinates of the new box. Normally this is randomly specified, but this allows
            it to be manually set. The coordinates are the absolute coordinates between 0 and
            image height/width and are int32.

    Returns:
        The new image that will have the shifted box location in it along with the new box that
        new box that contains the new coordinates.
    """
    image_height = tf.cast(tf.shape(image)[0], tf.float32)
    image_width = tf.cast(tf.shape(image)[1], tf.float32)

    def clip_y(val):
        return tf.clip_by_value(val, 0, tf.cast(image_height, tf.int32)-1)

    def clip_x(val):
        return tf.clip_by_value(val, 0, tf.cast(image_width, tf.int32)-1)

    # Convert box to pixel coordinates
    min_y = tf.cast(image_height * box[0], tf.int32)
    min_x = tf.cast(image_width * box[1], tf.int32)
    max_y = clip_y(tf.cast(image_height * box[2], tf.int32))
    max_x = clip_x(tf.cast(image_width * box[3], tf.int32))

    box_height, box_width = max_y - min_y + 1, max_x - min_x + 1

    image_height = tf.cast(image_height, tf.int32)
    image_width = tf.cast(image_width, tf.int32)

    # Select the new min/max box ranges that are used for sampling
    # the new min x/y coordinates of the shifted box.
    minval_y = clip_y(min_y - tf.cast(pixel_scaling * tf.cast(box_height, tf.float32) * 0.5, tf.int32))
    maxval_y = clip_y(min_y + tf.cast(pixel_scaling * tf.cast(box_height, tf.float32) * 0.5, tf.int32))
    minval_x = clip_x(min_x - tf.cast(pixel_scaling * tf.cast(box_width, tf.float32) * 0.5, tf.int32))
    maxval_x = clip_x(min_x + tf.cast(pixel_scaling * tf.cast(box_width, tf.float32) * 0.5, tf.int32))

    # Sample and calculate the new unclipped min/max coordinates of the new box.
    if new_min_box_coords is None:
        unclipped_new_min_y = tf.random.uniform([], minval_y, maxval_y, dtype=tf.int32)
        unclipped_new_min_x = tf.random.uniform([], minval_x, maxval_x, dtype=tf.int32)
    else:
        unclipped_new_min_y, unclipped_new_min_x = (clip_y(new_min_box_coords[0]),
                                                    clip_x(new_min_box_coords[1]))

    unclipped_new_max_y = unclipped_new_min_y + box_height - 1
    unclipped_new_max_x = unclipped_new_min_x + box_width - 1

    # Determine if any of the new box was shifted outside the current image.
    # This is used for determining if any of original box content should be
    # discarded.
    new_min_y, new_min_x, new_max_y, new_max_x = (clip_y(unclipped_new_min_y),
                                                  clip_x(unclipped_new_min_x),
                                                  clip_y(unclipped_new_max_y),
                                                  clip_x(unclipped_new_max_x))
    shifted_min_y = (new_min_y - unclipped_new_min_y) + min_y
    shifted_max_y = max_y - (unclipped_new_max_y - new_max_y)
    shifted_min_x = (new_min_x - unclipped_new_min_x) + min_x
    shifted_max_x = max_x - (unclipped_new_max_x - new_max_x)

    # Create the new box tensor by converting pixel integer values to floats.
    new_box = tf.stack([tf.cast(new_min_y, tf.float32) / tf.cast(image_height, tf.float32),
                        tf.cast(new_min_x, tf.float32) / tf.cast(image_width, tf.float32),
                        tf.cast(new_max_y, tf.float32) / tf.cast(image_height, tf.float32),
                        tf.cast(new_max_x, tf.float32) / tf.cast(image_width, tf.float32)])

    # Copy the contents in the box and fill the old box locations with gray(128).
    box_content = image[shifted_min_y:shifted_max_y+1, shifted_min_x:shifted_max_x+1, :]

    def mask_and_add_image(min_y_, min_x_, max_y_, max_x_, mask, content_tensor, image_):
        """Applies mask to box region in image then adds content_tensor to it."""
        mask = tf.pad(mask,
                      [[min_y_, (image_height-1)-max_y_],
                       [min_x_, (image_width-1)-max_x_], [0, 0]],
                      constant_values=1)
        content_tensor = tf.pad(content_tensor,
                                [[min_y_, (image_height - 1) - max_y_],
                                 [min_x_, (image_width - 1) - max_x_], [0, 0]],
                                constant_values=0)

        return image_ * mask + content_tensor

    # Zero out original box location
    mask = tf.zeros_like(image)[min_y:max_y+1, min_x:max_x+1, :]
    grey_tensor = tf.zeros_like(mask) + replace[0]
    image = mask_and_add_image(min_y, min_x, max_y, max_x, mask, grey_tensor, image)

    # Fill in box content to new box location.
    mask = tf.zeros_like(box_content)
    image = mask_and_add_image(new_min_y, new_min_x, new_max_y, new_max_x, mask, box_content, image)

    return image, new_box


def translate_x_only_boxes(image, boxes, prob, pixels, replace):
    """Apply translate_x to each box in image with probability prob."""
    func_changes_box = False
    prob = scale_box_only_op_probability(prob)

    return apply_multi_box_augmentation_wrapper(
        image, boxes, prob, translate_x, func_changes_box, pixels, replace)


def translate_y_only_boxes(image, boxes, prob, pixels, replace):
    """Apply translate_y to each box in image with probability prob."""
    func_changes_box = False
    prob = scale_box_only_op_probability(prob)

    return apply_multi_box_augmentation_wrapper(
        image, boxes, prob, translate_y, func_changes_box, pixels, replace)

