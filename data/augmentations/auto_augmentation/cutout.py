import tensorflow as tf
from data.auto_augmentation import scale_box_only_op_probability
from data.auto_augmentation import apply_multi_box_augmentation_wrapper


def cutout(image, size, replace=0):
    """Apply cutout to image.

    This operation applies a (2*padding X 2*padding) mask of zeros to
    a random location within `img`. The pixel values filled in will be
    of the value `replace`. The located where the mask will be applied
    is randomly chose uniformly over the whole image.

    Args:
        image: An image Tensor of type uint8.
        size: Specifies how big the zero mask that will be generated
            is that is applied to the image. The mask will be of size
            (2*size X 2*size).
        replace: what pixel value to fill in the image in the area that
            has the cutout mask applied to it.
    Returns:
        An image Tensor that is of type uint8.
    """
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    # Sample the center location in the image where the zero mask will be applied.
    cutout_center_y = tf.random.uniform([], 0, image_height, dtype=tf.int32)
    cutout_center_x = tf.random.uniform([], 0, image_width, dtype=tf.int32)

    top_padding = tf.maximum(0, cutout_center_y - size)
    bottom_padding = tf.maximum(0, image_height - cutout_center_y - size)
    left_padding = tf.maximum(0, cutout_center_x - size)
    right_padding = tf.maximum(0, image_width - cutout_center_x - size)

    cutout_shape = [image_height - (top_padding + bottom_padding),
                    image_width - (left_padding + right_padding)]
    padding = [[top_padding, bottom_padding], [left_padding, right_padding]]
    mask = tf.pad(tf.zeros(cutout_shape, dtype=image.dtype), padding, constant_values=1)
    mask = tf.tile(tf.expand_dims(mask, -1), [1, 1, 3])

    image = tf.where(tf.equal(mask, 0), tf.ones_like(image, dtype=image.dtype)*replace, image)

    return image


def _cutout_inside_box(image, box, pad_fraction):
    """Generates cutout mask and the mean pixel value of the box.

    First a location is randomly chosen within the image as the center
    where the cutout mask will be applied. Note this can be towards the
    boundaries of the image, so the full cutout mask may not be applied.

    Args:
        image: 3D uint8 Tensor.
        box: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
            of type float that represents the normalized coordinates
            between 0 and 1.
        pad_fraction: Float that specifies how large the cutout mask should
            be in reference to the size of the original box. If pad_fraction
            is 0.25, then the cutout mask will be of shape (0.25 * box_height, 0.25 * box_width).

    Returns:
        A tuple. First element is a tensor of the same shape as image where each
        element is either a 1 or 0 that is used to determine where the image
        will have cutout applied. The second element is the mean of the pixels
        in the image where the box is located.
    """
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    # Transform from shape [1, 4] to [4].
    box = tf.squeeze(box)

    min_y = tf.cast(tf.cast(image_height, tf.float32) * box[0], tf.int32)
    min_x = tf.cast(tf.cast(image_width, tf.float32) * box[1], tf.int32)
    max_y = tf.cast(tf.cast(image_height, tf.float32) * box[2], tf.int32)
    max_x = tf.cast(tf.cast(image_width, tf.float32) * box[3], tf.int32)

    # Calculate the mean pixel values in the bounding box, which will be
    # used to fill the cutout region.
    mean = tf.reduce_mean(image[min_y:max_y+1, min_x:max_x:1], axis=[0, 1])

    # Cutout mask will be size pad_size_height * 2 by pad_size_width * 2
    # if the region lies entirely within the box.
    box_height = max_y - min_y + 1
    box_width = max_x - min_x + 1
    pad_size_height = tf.cast(pad_fraction * (box_height / 2), tf.int32)
    pad_size_width = tf.cast(pad_fraction * (box_width / 2), tf.int32)

    # Sample the center location in the image where the zero mask will
    # be applied.
    cutout_center_height = tf.random.uniform([], min_y, max_y+1, tf.int32)
    cutout_center_width = tf.random.uniform([], min_x, max_x+1, tf.int32)

    lower_pad = tf.maximum(0, cutout_center_height - pad_size_height)
    upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size_height)
    left_pad = tf.maximum(0, cutout_center_width - pad_size_width)
    right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size_width)

    cutout_shape = [image_height - (lower_pad + upper_pad),
                    image_width - (left_pad + right_pad)]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]

    mask = tf.pad(tf.zeros(cutout_shape, dtype=image.dtype), padding_dims, constant_values=1)
    mask = tf.expand_dims(mask, 2)
    mask = tf.tile(mask, [1, 1, 3])

    return mask, mean


def box_cutout(image, boxes, pad_fraction, replace_with_mean):
    """Applies cutout to the image according to box information.

    This is a cutout variant that using box information to make more informed
    decisions on where to place the cutout mask.

    Args:
        image: 3D uint8 Tensor.
        boxes: 2D Tensor that is a list of the boxes in the image. Each box
            has 4 elements (min_y, min_x, max_y, max_x) of type float with values
            between [0, 1].
        pad_fraction: Float that specifies how large the cutout mask should be in
            in reference to the size of the original box. If pad_fraction is 0.25,
            then the cutout mask will be of shape
            (0.25 * box height, 0.25 * box width).
        replace_with_mean: Boolean that specified what value should be filled in
            where the cutout mask is applied. Since the incoming image will be of
            uint8 and will not have had any mean normalization applied, by default
            we set the value to be 128. If replace_with_mean is True then we find
            the mean pixel values across the channel dimension and use those to fill
            in where the cutout mask is applied.

    Returns:
        A tuple. First element is a tensor of the same shape as image that has
        cutout applied to it. Second element is the boxes that were passed in
        that will be unchanged.
    """
    def apply_box_cutout(img, boxes, fraction):
        """Applies cutout to a single bounding box within image."""
        # Choose a single bounding box to apply cutout to.
        random_index = tf.random.uniform([], maxval=tf.shape(boxes)[0], dtype=tf.int32)
        # Select the corresponding box and apply cutout.
        chosen_box = tf.gather(boxes, random_index)

        mask, mean = _cutout_inside_box(img, chosen_box, fraction)

        # When applying cutout we either set the pixel value to 128 or to
        # the mean value inside the box.
        replace = mean if replace_with_mean else 128

        # Apply the cutout mask to the image. where the mask is 0 we will fill
        # it with `replace`.
        img = tf.where(tf.equal(mask, 0),
                       tf.cast(tf.ones_like(img, dtype=img.dtype) * replace, dtype=image.dtype),
                       img)

        return img

    # Check to see if there are boxes, if so then apply box cutout.
    image = tf.cond(tf.equal(tf.size(boxes), 0),
                    lambda: image,
                    lambda: apply_box_cutout(image, boxes, pad_fraction))

    return image, boxes


def cutout_only_boxes(image, boxes, prob, pad_size, replace):
    """Apply cutout to each box in the image with probability prob."""
    func_changes_box = False
    prob = scale_box_only_op_probability(prob)

    return apply_multi_box_augmentation_wrapper(
        image, boxes, prob, cutout, func_changes_box, pad_size, replace)
