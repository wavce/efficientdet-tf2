import tensorflow as tf


# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
MAX_LEVEL = 10.


# Represents an invalid bounding box that is used for checking for padding
# lists of bounding box coordinates for a few augmentation operations
INVALID_BOX = [[-1.0, -1.0, -1.0, -1.0]]


CUTOUT_MAX_PAD_FRACTION = 0.75
CUTOUT_BOX_REPLACE_WITH_MEAN = False
CUTOUT_CONST = 100
TRANSLATE_CONST = 250
CUTOUT_BOX_CONST = 50
TRANSLATE_BOX_CONST = 120
REPLACE_VALUE = 128


def blend(image1, image2, factor):
    """Blend image1 and image2 using "factor".

        Factor can be above 0.0.  A value of 0.0 means only image1 is used.
        A value of 1.0 means only image2 is used.  A value between 0.0 and
        1.0 means we linearly interpolate the pixel values between the two
        images.  A value greater than 1.0 "extrapolates" the difference
        between the two pixel values, and we clip the results to values
        between 0 and 255.

        Args:
            image1: An image Tensor of type uint8.
            image2: An image Tensor of type uint8.
            factor: A floating point value above 0.0.
        Returns:
            A blended image Tensor of type uint8.
    """
    if tf.equal(factor, 0.0):
        return tf.cast(image1, tf.uint8)
    if tf.equal(factor, 1.0):
        return tf.cast(image2, tf.uint8)

    image2 = tf.cast(image2, tf.float32)
    image1 = tf.cast(image1, tf.float32)
    difference = image2 - image1
    scaled = difference * factor

    # Do addition in float.
    tmp = image1 + scaled

    # Interpolate
    if tf.greater(factor, 0.) and tf.less(factor, 1.):
        return tf.cast(tmp, tf.uint8)

    # Extrapolation
    return tf.cast(tf.clip_by_value(tmp, 0, 255), tf.uint8)


def clip_box(min_y, min_x, max_y, max_x):
    """Clip bounding box coordinates between 0 and 1.

    Args:
        min_y: Normalized box coordinate of type float between 0 and 1.
        min_x: Normalized box coordinate of type float between 0 and 1.
        max_y: Normalized box coordinate of type float between 0 and 1.
        max_x: Normalized box coordinate of type float between 0 and 1.

    Returns:
        Clipped coordinate values between 0 and 1.
    """
    min_y = tf.clip_by_value(min_y, 0.0, 1.0)
    min_x = tf.clip_by_value(min_x, 0.0, 1.0)
    max_y = tf.clip_by_value(max_y, 0.0, 1.0)
    max_x = tf.clip_by_value(max_x, 0.0, 1.0)

    return min_y, min_x, max_y, max_x


def check_box_area(min_y, min_x, max_y, max_x, delta=0.05):
    """Adjusts box coordinates to make sure the area is > 0.

    Args:
        min_y: Normalized box coordinate of type float between 0 and 1.
        min_x: Normalized box coordinate of type float between 0 and 1.
        max_y: Normalized box coordinate of type float between 0 and 1.
        max_x: Normalized box coordinate of type float between 0 and 1.
        delta: Float, this is used to create a gap of size 2 * delta between
            box min/max coordinates that are the same on the boundary.
            This prevents the box from having an area of zero.

    Returns:
        Tuple of new box coordinates between 0 and 1 that will now have a
        guaranteed area > 0.
    """
    height = max_y - min_y
    width = max_x - min_x

    def _adjust_box_boundaries(min_coord, max_coord):
        # Make sure max is never 0 and min is never 1.
        max_coord = tf.maximum(max_coord, 0.0+delta)
        min_coord = tf.minimum(min_coord, 1.0-delta)

        return min_coord, max_coord

    min_y, max_y = tf.cond(tf.equal(height, 0.0),
                           lambda: _adjust_box_boundaries(min_y, max_y),
                           lambda: (min_y, max_y))
    min_x, max_x = tf.cond(tf.equal(width, 0.0),
                           lambda: _adjust_box_boundaries(min_x, max_x),
                           lambda: (min_x, max_x))

    return min_y, min_x, max_y, max_x


def scale_box_only_op_probability(prob):
    """Reduce the probability of the box-only operation.

    Probability is reduced so that we do not distort the content of too many
    bounding boxes that are close to each other. The value of 3.0 was a chosen
    hyper parameter when designing the autoaugment algorithm that we found
    empirically to work well.

    Args:
        prob: Float that is the probability of applying the box-only operation.
    Returns:
        Reduced probability.
    """
    return prob / 3.0


def apply_box_augmentation(image, box, augmentation_func, *args):
    """Applies augmentation_func to the subsection of image indicated by box.

    Args:
        image: 3D uint8 Tensor.
        box: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
            of type float that represents the normalized coordinates between 0 and 1.
        augmentation_func: Augmentation function that will be applied to the
            subsection of image.
        *args: Additional parameters that will be passed into augmentation_func
            when it is called.

    Returns:
        A modified version of image, where the box location in the image will
        have `augmentation_func applied to it.
    """
    image_height = tf.cast(tf.shape(image)[0], tf.float32)
    image_width = tf.cast(tf.shape(image)[1], tf.float32)

    min_y = tf.cast(image_height * box[0], tf.int32)
    min_x = tf.cast(image_width * box[1], tf.int32)
    max_y = tf.cast(image_height * box[2], tf.int32)
    max_x = tf.cast(image_width * box[3], tf.int32)

    image_height = tf.cast(image_height, tf.int32)
    image_width = tf.cast(image_width, tf.int32)

    # Clip to be sure the mas values do not fall out of range.
    max_y = tf.minimum(max_y, image_height - 1)
    max_x = tf.minimum(max_x, image_width - 1)

    # Get the sub-tensor that is the image within the bounding box region.
    box_content = image[min_y:max_y + 1, min_x:max_x + 1, :]

    # Apply the augmentation function to the box portion of the image.
    augmented_box_content = augmentation_func(box_content, *args)

    # Pad the augmented_box_content and the mask to match the shape of original image
    augmented_box_content = tf.pad(augmented_box_content,
                                   [[min_y, (image_height - 1) - max_y],
                                    [min_x, (image_width - 1) - max_x],
                                    [0, 0]])
    # Create a mask that will be used to zero out a part of the original image.
    mask_tensor = tf.zeros_like(box_content)

    mask_tensor = tf.pad(mask_tensor,
                         [[min_y, (image_height-1)-max_y],
                          [min_x, (image_width-1)-max_x],
                          [0, 0]],
                         constant_values=1)

    # Replace the old box content with the new augmented content.
    image = image * mask_tensor + augmented_box_content

    return image


def concat_box(box, boxes):
    """Helper function that concates box to boxes along the first dimension."""

    # Note if all elements in boxes are -1 (_INVALID_BOX), then this means
    # we discard boxes and start the boxes Tensor with the current box.
    boxes_sum_check = tf.reduce_sum(boxes)
    box = tf.expand_dims(box, 0)
    # This check will be true when it is an _INVALID_BOX
    boxes = tf.cond(tf.equal(boxes_sum_check, -4.0),
                     lambda: box,
                     lambda: tf.concat([boxes, box], 0))

    return boxes


def apply_box_augmentation_wrapper(image, box, new_boxes, prob, augmentation_func, func_changes_box, *args):
    """Applies _apply_box_augmentation with probability prob.

    Args:
        image: 3D uint8 Tensor.
        box: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
            of type float that represents the normalized coordinates between 0 and 1.
        new_boxes: 2D Tensor that is a list of the boxes in the image after they
            have been altered by aug_func. These will only be changed when
            func_changes_box is set to true. Each box has 4 elements
            (min_y, min_x, max_y, max_x) of type float that are the normalized
            box coordinates between 0 and 1.
        prob: Float that is the probability of applying _apply_box_augmentation.
        augmentation_func: Augmentation function that will be applied to the
            subsection of image.
        func_changes_box: Boolean. Does augmentation_func return box in addition
            to image.
        *args: Additional parameters that will be passed into augmentation_func
            when it is called.

    Returns:
        A tuple. Fist element is a modified version of image, where the box
        location in the image will have augmentation_func applied to it if it is
        chosen to be called with probability `prob`. The second element is a
        Tensor of Tensors of length 4 that will contain the altered box after
        applying augmentation_func.
    """
    should_apply_op = tf.cast(tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)

    if func_changes_box:
        augmented_image, box = tf.cond(should_apply_op,
                                       lambda: augmentation_func(image, box, *args),
                                       lambda: (image, box))
    else:
        augmented_image = tf.cond(should_apply_op,
                                  lambda: apply_box_augmentation(image, box, augmentation_func, *args),
                                  lambda: image)

    new_boxes = concat_box(box, new_boxes)

    return augmented_image, new_boxes


def apply_multi_box_augmentation(image, boxes, prob, aug_func, func_changes_box, *args):
    """Applies aug_func to the image for each box in boxes.

    Args:
        image: 3D uint8 Tensor.
        boxes: 2D Tensor that is a list of the boxes in the image. Each box
            has 4 elements (min_y, min_x, max_y, max_x) of type float.
        prob: Float that is the probability of applying aug_func to a specific
            bounding box within the image.
        aug_func: Augmentation function that will be applied to the
            subsections of image indicated by the box values in boxes.
        func_changes_box: Boolean. Does augmentation_func return box in addition
            to image.
        *args: Additional parameters that will be passed into augmentation_func
            when it is called.

    Returns:
        A modified version of image, where each box location in the image will
        have augmentation_func applied to it if it is chosen to be called with
        probability prob independently across all boxes. Also the final
        boxes are returned that will be unchanged if func_changes_box is set to
        false and if true, the new altered ones will be returned.
    """
    # Will keep track of the new altered boxes after aug_func is repeatedly
    # applied. The -1 values are a dummy value and this first Tensor will be
    # removed upon appending the first real box.
    new_boxes = tf.constant(INVALID_BOX)

    # If the boxes are empty, then just give it _INVALID_BOX. The result
    # will be thrown away.
    boxes = tf.cond(tf.equal(tf.size(boxes), 0),
                    lambda: tf.constant(INVALID_BOX),
                    lambda: boxes)
    boxes = tf.ensure_shape(boxes, [None, 4])

    # pylint: disable=g-long-lambda
    # pylint: disable=line-too-long
    wrapped_aug_func = lambda _img, _box, _new_boxes: apply_box_augmentation_wrapper(
        _img, _box, _new_boxes, prob, aug_func, func_changes_box, *args)
    # pylint:enable=g-long-lambda
    # pylint:enable=line-too-long

    # Setup the while_loop.
    num_boxes = tf.shape(boxes)[0]  # We loop until we go over all boxes.
    idx = tf.constant(0)  # Counter for the while loop.

    # Conditional function when to end the loop once we go over all boxes
    # images_and_boxes contain (_image, _new_boxes)
    cond = lambda _idx, _image_and_boxes: tf.less(_idx, num_boxes)

    # Shuffle the boxes so that the augmentation order is not deterministic
    # if we are not changing the boxes with aug_func.
    if not func_changes_box:
        loop_boxes = tf.random.shuffle(boxes)
    else:
        loop_boxes = boxes

    # Main function of while_loop where we repeatedly apply augmentation on
    # the boxes in the image.
    # pylint: disable=g-long-lambda
    body = lambda _idx, _image_and_boxes: [_idx + 1, wrapped_aug_func(
        _image_and_boxes[0], loop_boxes[idx], _image_and_boxes[1])]
    # pylint: enable=g-long-lambda
    _, (image, new_boxes) = tf.while_loop(cond, body, [idx, (image, new_boxes)],
                                          shape_invariants=[idx.get_shape(),
                                                            (image.get_shape(), tf.TensorShape([None, 4]))])

    # Either return the altered boxes or the original ones depending on if
    # we altered them in anyway.
    if func_changes_box:
        final_boxes = new_boxes
    else:
        final_boxes = boxes

    return image, final_boxes


def apply_multi_box_augmentation_wrapper(image, boxes, prob, aug_func, func_changes_box, *args):
    """Checks to be sure num boxes > 0 before calling inner function."""
    num_boxes = tf.shape(boxes)[0]
    image, boxes = tf.cond(
        tf.equal(num_boxes, 0),
        lambda: (image, boxes),
        # pylint:disable=g-long-lambda
        lambda: apply_multi_box_augmentation(
            image, boxes, prob, aug_func, func_changes_box, *args))
    # pylint:enable=g-long-lambda

    return image, boxes


def wrap(image):
    """Returns `image` with an extra channel set to all ones."""
    shape = tf.shape(image)

    extended_channel = tf.ones([shape[0], shape[1], 1], image.dtype)
    extended = tf.concat([image, extended_channel], 2)

    return extended


def unwrap(image, replace):
    """Unwraps an image produced by wrap.

    Where there is a 0 in the last channel for every spatial position,
    the rest of the three channels in that spatial dimension are grayed
    (set ot 128). Operations like translate and shear on a wrapped Tensor
    will leave 0s in empty locations. Some transformations look at the
    intensity of values to do preprocessing, and we want these empty pixels
    to assume the `average` value, rather than pure black.

    Args:
        image: A 3D image Tensor with 4 channels.
        replace: A one or three value 1D tensor to fill empty pixels.
    Returns:
        image: A 3D image Tensor with 3 channels.
    """
    image_shape = tf.shape(image)

    # Flatten the spatial dimensions.
    flattened_image = tf.reshape(image, [-1, image_shape[2]])

    # Find all pixels where the last channel is zero.
    alpha_channel = flattened_image[:, 3:]

    replace = tf.concat([replace, tf.ones([1], image.dtype)], 0)

    # Where the are zero, fill them in with `replace`.
    flattened_image = tf.where(tf.equal(alpha_channel, 0),
                               tf.ones_like(flattened_image, dtype=image.dtype) * replace,
                               flattened_image)
    image = tf.reshape(flattened_image, image_shape)
    image = tf.slice(image, [0, 0, 0], [image_shape[0], image_shape[1], 3])

    return image
