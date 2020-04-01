import math
import numpy as np
import tensorflow as tf
from skimage import transform
# from data.auto_augmentation.augmentation import policy, resize_and_crop_image, compute_padded_size


def should_apply_op(prob):
    return tf.cast(tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)


def jaccard(boxes, patch):
    lt = tf.maximum(boxes[:, 0:2], patch[0:2])
    rb = tf.minimum(boxes[:, 2:4], patch[2:4])

    wh = tf.maximum(0.0, rb - lt)  # (n, m, 2)
    overlap = tf.reduce_prod(wh, axis=1)  # (n, m)
    box_areas = tf.reduce_prod(boxes[:, 2:4] - boxes[:, 0:2], axis=1)  # (n, m)
    patch_area = (patch[2] - patch[0]) * (patch[3] - patch[1])

    overlaps = overlap / (box_areas + patch_area - overlap)

    return overlaps


def is_box_center_in_patch(boxes, patch):
    box_ctr = (boxes[:, 0:2] + boxes[:, 2:4]) * 0.5
    flags = tf.logical_and(x=tf.greater(box_ctr, patch[0:2]),
                           y=tf.less(box_ctr, patch[2:4]))
    flags = tf.reduce_all(flags, axis=1)

    return flags


def clip_boxes_based_center(boxes, labels, patch):
    mask = is_box_center_in_patch(boxes, patch)

    clipped_boxes = tf.boolean_mask(boxes, mask)
    clipped_labels = tf.boolean_mask(labels, mask)
    mask = is_box_center_in_patch(boxes, patch)
    clipped_boxes = tf.boolean_mask(boxes, mask)

    patch_yx = tf.convert_to_tensor([patch[0], patch[1], patch[0], patch[1]], dtype=boxes.dtype)
    patch_hw = tf.convert_to_tensor([patch[2] - patch[0], patch[3] - patch[1]] * 2, boxes.dtype)
    clipped_boxes = tf.clip_by_value((clipped_boxes - patch_yx) / patch_hw, 0, 1)


    return clipped_boxes, clipped_labels


class FlipLeftToRight(object):
    def __init__(self, probability=0.5):
        self.prob = probability

    def _flip(self, image, boxes, labels):
        image = tf.image.flip_left_right(image)

        y1, x1, y2, x2 = tf.unstack(boxes, 4, 1)
        new_x1 = 1. - x2
        new_x2 = 1. - x1

        boxes = tf.stack([y1, new_x1, y2, new_x2], 1)

        return image, boxes, labels

    def __call__(self, image, boxes, labels):
        with tf.name_scope("flip_left_to_right"):
            return tf.cond(should_apply_op(self.prob),
                           lambda: self._flip(image, boxes, labels), 
                           lambda: (image, boxes, labels))


class SSDCrop(object):
    def __init__(self,
                 input_size,
                 patch_area_range=(0.3, 1.),
                 aspect_ratio_range=(0.5, 2.0),
                 min_overlaps=(0.1, 0.3, 0.5, 0.7, 0.9),
                 max_attempts=100,
                 **kwargs):
        self.input_size = input_size
        self.patch_area_range = patch_area_range
        self.aspect_ratio_range = aspect_ratio_range
        self.min_overlaps = tf.constant(min_overlaps, dtype=tf.float32)
        self.max_attempts = max_attempts

    def _random_overlaps(self):
        return tf.random.shuffle(self.min_overlaps)[0]

    def _random_crop(self, image, boxes, labels):
        image_size = tf.shape(image)
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
            image_size=image_size,
            bounding_boxes=tf.expand_dims(boxes, 0),
            min_object_covered=self._random_overlaps(),
            aspect_ratio_range=self.aspect_ratio_range,
            area_range=self.patch_area_range,
            max_attempts=self.max_attempts,
            use_image_if_no_bounding_boxes=True)

        patch = distort_bbox[0, 0]
        cropped_boxes, cropped_labels = clip_boxes_based_center(boxes, labels, patch)
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        cropped_image = tf.reshape(cropped_image, [bbox_size[0], bbox_size[1], 3])
        cropped_image = tf.image.resize(cropped_image, self.input_size)

        return cropped_image, cropped_boxes, cropped_labels

    def __call__(self, image, boxes, labels):
        with tf.name_scope("ssd_crop"):
            return self._random_crop(image, boxes, labels)


class DataAnchorSampling(object):
    def __init__(self,
                 input_size=(640, 640),
                 anchor_scales=(16, 32, 64, 128, 256, 512),
                 overlap_threshold=0.7,
                 max_attempts=50,
                 **Kwargs):
        self.anchor_scales = tf.reshape(tf.constant(anchor_scales, tf.float32), [-1])
        self.num_scales = tf.shape(self.anchor_scales, out_type=tf.int64)[0]
        self.input_size = input_size
        self.overlap_threshold = tf.convert_to_tensor(overlap_threshold, tf.float32)

        self.mean = tf.convert_to_tensor([104, 117, 123], dtype=tf.float32)
        self.max_size = 12000
        self.inf = 9999999
        self.max_attempts = max_attempts

    def _sampling(self, image, boxes, labels):
        image_shape = tf.shape(image)
        # image_size = tf.convert_to_tensor([image_shape[0],
        #                                    image_shape[1],
        #                                    image_shape[0],
        #                                    image_shape[1]], tf.float32)
        # boxes = boxes * image_size
        image_size = tf.cast(image_shape[0:2], tf.float32)
        height, width = image_size[0], image_size[1]
        box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * height * width
        box_areas = box_areas[box_areas > 0]

        if tf.less_equal(tf.size(box_areas), 0):
            image = tf.image.resize(image, self.input_size)
            return image, boxes, labels

        rand_box_idx = tf.random.uniform([], 0, tf.shape(box_areas)[0], dtype=tf.int32)
        rand_box_scale = tf.math.sqrt(box_areas[rand_box_idx])

        anchor_idx = tf.argmin(tf.math.abs(self.anchor_scales - rand_box_scale))
        anchor_idx_range = tf.minimum(anchor_idx + 1, self.num_scales) + 1
        target_anchor = tf.random.shuffle(self.anchor_scales[0:anchor_idx_range])[-1]
        ratio = target_anchor / rand_box_scale
        ratio *= (tf.math.pow(2., tf.random.uniform([], -1, 1, dtype=tf.float32)))

        if tf.greater(height * ratio * width * ratio, self.max_size * self.max_size):
            ratio = (self.max_size * self.max_size / (height * width)) ** 0.5
        else:
            ratio = ratio

        resizing_height = height * ratio
        resizing_width = width * ratio
        resizing_size = tf.convert_to_tensor([resizing_height, resizing_width], tf.int32)
        image = tf.image.resize(image, resizing_size)

        y1 = boxes[rand_box_idx, 0] * height
        x1 = boxes[rand_box_idx, 1] * width
        y2 = boxes[rand_box_idx, 2] * height
        x2 = boxes[rand_box_idx, 3] * width
        rand_box_w = x2 - x1 + 1
        rand_box_h = y2 - y1 + 1
        inp_h = tf.cast(self.input_size[0], tf.float32)
        inp_w = tf.cast(self.input_size[1], tf.float32)

        sample_boxes = tf.TensorArray(size=50, dtype=boxes.dtype)
        for i in tf.range(50):
            if tf.less(inp_w, tf.maximum(resizing_height, resizing_width)):
                if tf.less_equal(rand_box_w, inp_w):
                    offset_width = tf.random.uniform([], x1 + rand_box_w - inp_w, x1)
                else:
                    offset_width = tf.random.uniform([], x1, x1 + rand_box_w - inp_w)
                if tf.less_equal(rand_box_h, inp_h):
                    offset_height = tf.random.uniform([], y1 + rand_box_h - inp_h, y1)
                else:
                    offset_height = tf.random.uniform([], y1, y1 + rand_box_h - inp_h)
            else:
                offset_height = tf.random.uniform([], resizing_height - inp_h, 0)
                offset_width = tf.random.uniform([], resizing_width - inp_w, 0)
            offset_height = tf.math.floor(offset_height)
            offset_width = tf.math.floor(offset_width)

            patch = tf.convert_to_tensor([offset_height / height, 
                                          offset_width / width, 
                                          (offset_height + inp_h) / height, 
                                          (offset_width + inp_w) / width], tf.float32)
            in_patch = is_box_center_in_patch(boxes, patch)
            overlaps = jaccard(boxes, patch)

            if tf.logical_or(tf.reduce_any(in_patch), tf.greater_equal(tf.reduce_max(overlaps), 0.7)):
                sample_boxes = sample_boxes.write(i, patch)
            else:
                continue
        
        sample_boxes = sample_boxes.stack()
        if tf.greater(tf.size(sample_boxes), 0):
            choice_patch = tf.random.shuffle(sample_boxes)[0]
            current_boxes, current_labels = clip_boxes_based_center(boxes, labels, choice_patch)

            if tf.logical_or(tf.less(choice_patch[0], 0), tf.less(choice_patch[1], 0)):
                if tf.greater_equal(choice_patch[0], 0):
                    top_padding = tf.zeros([], tf.int32)
                    offset_height = tf.cast(choice_patch[0], tf.int32)
                else:
                    # new_img_width = resizing_width - choice_patch[0]
                    top_padding = tf.cast(-1. * choice_patch[0], tf.int32)
                    offset_height = tf.zeros([], tf.int32)
                if tf.greater_equal(choice_patch[1], 0):
                    left_padding = tf.zeros([], tf.int32)
                    offset_width = tf.cast(choice_patch[1], tf.int32)
                else:
                    # new_img_height = resizing_height - choice_patch[1]
                    left_padding = tf.cast(-1. * choice_patch[1], tf.int32)
                    offset_width = tf.zeros([], tf.int32)

                # bottom_padding = tf.maximum(tf.cast(inp_h, tf.int32) - top_padding - resizing_size[0], 0)
                # right_padding = tf.maximum(tf.cast(inp_w, tf.int32) - left_padding - resizing_size[1], 0)
            else:
                left_padding = tf.zeros([], tf.int32)
                top_padding = tf.zeros([], tf.int32)
                offset_height = tf.cast(choice_patch[0], tf.int32)
                offset_width = tf.cast(choice_patch[1], tf.int32)

            bottom_padding = tf.minimum(resizing_size[0] - tf.cast(inp_h, tf.int32) - offset_height, 0) * -1
            right_padding = tf.minimum(resizing_size[1] - tf.cast(inp_w, tf.int32) - offset_width, 0) * -1
            target_height = tf.cast(choice_patch[2] - choice_patch[0], tf.int32)
            target_width = tf.cast(choice_patch[3] - choice_patch[1], tf.int32)

            # if tf.logical_or(choice_patch[0] + choice_patch[2] <= 2, choice_patch[1] + choice_patch[3] <= 2):
                # tf.print(choice_patch, [offset_height, offset_width, target_height, target_width],
                        #  [top_padding, bottom_padding, left_padding, right_padding], resizing_size, tf.shape(image))
            
            if tf.logical_and(target_width > 0, target_height > 0):
                padded_image = tf.pad(image,
                                  paddings=[[top_padding, bottom_padding], [left_padding, right_padding], [0, 0]],
                                  constant_values=128)
                cropped_image = tf.image.crop_to_bounding_box(image=padded_image,
                                                              offset_height=offset_height,
                                                              offset_width=offset_width,
                                                              target_height=target_height,
                                                              target_width=target_width)

                return cropped_image, current_boxes, current_labels
            else:
                image = tf.image.resize(image, self.input_size)
                return image, boxes, labels
        else:
            image = tf.image.resize(image, self.input_size)
            return image, boxes, labels

    def __call__(self, image, boxes, labels):
        with tf.name_scope("data_anchor_sampling"):
            return self._sampling(image, boxes, labels)


class RandomDistortColor(object):
    def __init__(self,
                 brightness=32./255.,
                 min_saturation=0.5,
                 max_saturation=1.5,
                 hue=0.2,
                 min_contrast=0.5,
                 max_contrast=1.5,
                 **Kwargs):
        self.brightness = brightness
        self.min_saturation = min_saturation
        self.max_saturation = max_saturation
        self.hue = hue
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast

    def _distort_color0(self, image):
        image = tf.image.random_brightness(image, max_delta=self.brightness)
        image = tf.image.random_saturation(image, lower=self.min_saturation, upper=self.max_saturation)
        image = tf.image.random_hue(image, max_delta=self.hue)
        image = tf.image.random_contrast(image, lower=self.min_contrast, upper=self.max_contrast)

        return image

    def _distort_color1(self, image):
        image = tf.image.random_saturation(image, lower=self.min_saturation, upper=self.max_saturation)
        image = tf.image.random_brightness(image, max_delta=self.brightness)
        image = tf.image.random_contrast(image, lower=self.min_contrast, upper=self.max_contrast)
        image = tf.image.random_hue(image, max_delta=self.hue)

        return image
    
    def _distort_color2(self, image):
        image = tf.image.random_contrast(image, lower=self.min_contrast, upper=self.max_contrast)
        image = tf.image.random_hue(image, max_delta=self.hue)
        image = tf.image.random_brightness(image, max_delta=self.brightness)
        image = tf.image.random_saturation(image, lower=self.min_saturation, upper=self.max_saturation)

        return image
    
    def _distort_color3(self, image):
        image = tf.image.random_hue(image, max_delta=self.hue)
        image = tf.image.random_saturation(image, lower=self.min_saturation, upper=self.max_saturation)
        image = tf.image.random_contrast(image, lower=self.min_contrast, upper=self.max_contrast)
        image = tf.image.random_brightness(image, max_delta=self.brightness)

        return image

    def __call__(self, image, boxes, labels):
        with tf.name_scope("distort_color"):
            rand_int = tf.random.uniform([], 0, 4, tf.int32)
            if rand_int == 0:
                image = self._distort_color0(image)
            elif rand_int == 1:
                image = self._distort_color1(image)
            elif rand_int == 2:
                image = self._distort_color2(image)
            else:
                image = self._distort_color3(image)
            
            image = tf.minimum(tf.maximum(image, 0), 255)

            return image, boxes, labels


class SFDetCrop(object):
    def __init__(self, input_size, **kwargs):
        self.input_size = input_size
    
    def _crop(self, image, boxes, labels):
        img_size = tf.cast(tf.shape(image)[0:2], tf.float32)

        min_size = tf.reduce_min(img_size)
        scales = [min_size]
        scales += [tf.random.uniform([], 0.3, 1.) * min_size for _ in range(5)]
        scale = tf.random.shuffle(scales)[-1]        
        
        offset_height =  tf.random.uniform([], 0, tf.cast((img_size[0] - scale) * 0.5, tf.int32) + 1, tf.int32)
        offset_width = tf.random.uniform([], 0, tf.cast((img_size[1] - scale) * 0.5, tf.int32) + 1, tf.int32)
        target_scale = tf.cast(scale, tf.int32)

        cropped_image = tf.image.crop_to_bounding_box(image=image,
                                                      offset_height=offset_height,
                                                      offset_width=offset_width,
                                                      target_height=target_scale,
                                                      target_width=target_scale)                       

        patch = tf.convert_to_tensor([offset_height, 
                                      offset_width, 
                                      offset_height + target_scale, 
                                      offset_width + target_scale], tf.float32)
        patch /= [img_size[0], img_size[1], img_size[0], img_size[1]]
        
        cropped_boxes, cropped_labels = clip_boxes_based_center(boxes, labels, patch) 

        cropped_image = tf.image.resize(cropped_image, self.input_size)

        return cropped_image, cropped_boxes, cropped_labels
    
    def call(self, image, boxes, labels):
        cropped_image = image
        cropped_boxes = boxes
        for _ in tf.range(50):
            cropped_image, cropped_boxes, cropped_labels = self._crop(image, boxes, labels)

            if tf.greater(tf.shape(cropped_boxes)[0], 0):
                break
        
        return cropped_image, cropped_boxes, cropped_labels
    
    def __call__(self, image, boxes, labels):
        with tf.name_scope("sfdet_crop"):
            return self.call(image, boxes, labels)


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


class Rotate(object):
    def __init__(self, min_angle=-45, max_angle=45, probability=1.0):
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.prob = probability

    def rotate(self, image, angle, replace):
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
        return tf.reshape(_rotate(image, radians, replace), tf.shape(image))

    def _rotate_box(self, box, label, image_shape, angle):
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
        min_y = -tf.cast(box[0] - image_height * 0.5, tf.int32)
        min_x = tf.cast(box[1] - image_width * 0.5, tf.int32)
        max_y = -tf.cast(box[2] - image_height * 0.5, tf.int32)
        max_x = tf.cast(box[3] - image_width * 0.5, tf.int32)

        coordinates = tf.stack([[min_y, min_x], [min_y, max_x], [max_y, min_x], [max_y, max_x]])
        coordinates = tf.cast(coordinates, tf.float32)
        # Rotate the coordinates according to the rotation matrix clockwise if
        # radians is positive, else negative.
        rotation_matrix = tf.stack([[tf.math.cos(radians), tf.math.sin(radians)],
                                    [-tf.math.sin(radians), tf.math.cos(radians)]])
        new_coords = tf.cast(tf.matmul(rotation_matrix, tf.transpose(coordinates)), tf.int32)
        # Find min/max values and convert them back to normalized 0-1 floats.
        min_y = -(tf.cast(tf.reduce_max(new_coords[0, :]), tf.float32) - image_height * 0.5)
        min_x = tf.cast(tf.reduce_min(new_coords[1, :]), tf.float32) + image_width * 0.5
        max_y = -(tf.cast(tf.reduce_min(new_coords[0, :]), tf.float32) - image_height * 0.5)
        max_x = tf.cast(tf.reduce_max(new_coords[1, :]), tf.float32) + image_width * 0.5

        # Clip the boxes to be sure the fall between [0, 1].
        min_y = tf.clip_by_value(min_y, 0, image_height)
        min_x = tf.clip_by_value(min_x, 0, image_width)
        max_y = tf.clip_by_value(max_y, 0, image_height)
        max_x = tf.clip_by_value(max_x, 0, image_width)

        h = max_y - min_y
        w = max_x - min_x
        min_y += (0.06 * h)
        min_x += (0.06 * w)
        max_y -= (0.06 * h)
        max_x -= (0.06 * w)
        
        rotated_box = tf.stack([min_y, min_x, max_y, max_x], -1)
        if tf.logical_and(tf.logical_and((min_x + max_x) * 0.5 > 0, 
                                         (min_x + max_x) * 0.5 < image_width),
                          tf.logical_and((min_y + max_y) * 0.5 > 0,
                                         (min_y + max_y) * 0.5 < image_height)):
            rotated_box = rotated_box
            rotated_label = label
        else:
            rotated_box = tf.zeros_like(rotated_box)
            rotated_label = tf.zeros_like(label)

        return rotated_boxes, rotated_label

    def rotate_with_boxes(self, image, boxes, labels, angle, replace):
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
        image = self.rotate(image, angle, replace)
        image = tf.cast(image, tf.float32)

        # Convert box coordinates to pixel values
        image_shape = tf.shape(image)[0:2]
        boxes, labels = tf.map_fn(lambda box, label: self._rotate_box(box, label, image_shape, angle), 
                                  elems=(boxes, labels), 
                                  dtype=[boxes.dtype, labels.dtype])

        return image, boxes, labels
    
    def __call__(self, image, boxes, labels):
        with tf.name_scope("rotate"):
            angle = tf.random.uniform([], self.min_angle, self.max_angle)
            return tf.cond(should_apply_op(self.prob),
                           lambda: self.rotate_with_boxes(image, boxes, labels, angle, replace=125),
                           lambda: (image, boxes, labels))


# class AutoAugmentation(object):
#     def __init__(self, input_size, augmentation_name="v0"):
#         self.flip = FlipLeftToRight(0.5)

#         self.input_size = tf.convert_to_tensor(input_size, tf.int32)
#         self.probability = tf.convert_to_tensor(0.5, tf.float32)
#         self._policy = policy(augmentation_name)

#     def _process(self, image, boxes):
#         # if training:
#         #     image = tf.cast(image, tf.uint8)
#         #     image, boxes = self._policy(image, boxes)

#         image, boxes = resize_and_crop_image(image=image,
#                                              boxes=boxes,
#                                              desired_size=self.input_size,
#                                              padded_size=compute_padded_size(self.input_size, 128),
#                                              aug_scale_min=0.8,
#                                              aug_scale_max=1.2,
#                                              random_jittering=True)
#         image = tf.cast(image, tf.float32)

#         return image, boxes

#     def __call__(self, image, boxes):
#         return self._process(image, boxes)
