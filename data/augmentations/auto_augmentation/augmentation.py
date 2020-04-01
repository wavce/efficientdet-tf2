# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from data.auto_augmentation import *


def randomly_negate_tensor(tensor):
    """With 50% prob turn the tensor negative."""
    should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
    final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)

    return final_tensor


def rotate_level_to_arg(level):
    level = (level / MAX_LEVEL) * 30.
    return randomly_negate_tensor(level)


def shrink_level_to_arg(level):
    """Converts level to ratio by which we shrink the image content."""
    if level == 0:
        level = 1.
        return level  # if level is zero, do not shrink the image

    # Maximum shrinking ratio is 2.9.
    level = 2. / (MAX_LEVEL / level) + 0.9

    return level


def enhance_level_to_arg(level):
    level = (level / MAX_LEVEL) * 1.8 + 0.1

    return level


def shear_level_to_arg(level):
    level = (level / MAX_LEVEL) * 0.3

    # Flip level to negative with 50% chance.
    level = randomly_negate_tensor(level)

    return level


def translate_level_to_arg(level, translate_const):
    level = (level / MAX_LEVEL) * float(translate_const)

    # Flip level to negative with 50% chance.
    level = randomly_negate_tensor(level)

    return level


def bbox_cutout_level_to_arg(level, cutout_max_pad_fraction, cutout_bbox_replace_with_mean):
    cutout_pad_fraction = (level/MAX_LEVEL) * cutout_max_pad_fraction

    return (cutout_pad_fraction,
            cutout_bbox_replace_with_mean)


def should_apply_op(prob):
    return tf.cast(tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)


def policy_v0(image, boxes):
    """Autoaugment policy that was used in AutoAugment Detection Paper."""

    # Each tuple is an augmentation operation of the form
    # (operation, probability, magnitude). Each element in policy is a
    # sub-policy that will be applied sequentially on the image.

    i = tf.random.uniform([], 0, 5, dtype=tf.int32)
    if tf.equal(i, 0):  # [('TranslateXBox', 0.6, 4), ('Equalize', 0.8, 10)]
        if should_apply_op(0.6):
            image, boxes = translate_boxes(
                image, boxes, translate_level_to_arg(4, TRANSLATE_CONST), REPLACE_VALUE, True)
        if should_apply_op(0.8):
            image = equalize(image)
    elif tf.equal(i, 1):  # [('TranslateYOnlyBoxes', 0.2, 2), ('Cutout', 0.8, 8)]
        image, boxes = translate_y_only_boxes(
            image, boxes, 0.2, translate_level_to_arg(2, TRANSLATE_BOX_CONST), REPLACE_VALUE)
        if should_apply_op(0.8):
            image = cutout(image, int((8 / MAX_LEVEL) * CUTOUT_CONST), REPLACE_VALUE)
    elif tf.equal(i, 2):  # [('Sharpness', 0.0, 8), ('ShearXBox', 0.4, 0)]
        if should_apply_op(0.0):
            image = sharpness(image, enhance_level_to_arg(8))
        if should_apply_op(0.4):
            image, boxes = shear_with_boxes(
                image, boxes, shear_level_to_arg(0), REPLACE_VALUE, True)
    elif tf.equal(i, 3):  # [('ShearYBox', 1.0, 2), ('TranslateYOnlyBoxes', 0.6, 6)]
        if should_apply_op(1.0):
            image, boxes = shear_with_boxes(
                image, boxes, shrink_level_to_arg(2), REPLACE_VALUE, False)
        image, boxes = translate_y_only_boxes(
            image, boxes, 0.6, translate_level_to_arg(6, TRANSLATE_BOX_CONST), REPLACE_VALUE)
    else:  # [('RotateBox', 0.6, 10), ('Color', 1.0, 6)]
        image, boxes = rotate_with_boxes(image, boxes, rotate_level_to_arg(10), REPLACE_VALUE)
        if should_apply_op(1.):
            image = color(image, enhance_level_to_arg(6))

    return image, boxes


def policy_v1(image, boxes):
    """Autoaugment policy that was used in AutoAugment Detection Paper."""

    # Each tuple is an augmentation operation of the form
    # (operation, probability, magnitude). Each element in policy is a
    # sub-policy that will be applied sequentially on the image.

    i = tf.random.uniform([], 0, 20, dtype=tf.int32)
    if tf.equal(i, 0):  # [('TranslateXBox', 0.6, 4), ('Equalize', 0.8, 10)]
        if should_apply_op(0.6):
            image, boxes = translate_boxes(
                image, boxes, translate_level_to_arg(4, TRANSLATE_CONST), REPLACE_VALUE, True)
        if should_apply_op(0.8):
            image = equalize(image)
    elif tf.equal(i, 1):  # [('TranslateYOnlyBoxes', 0.2, 2), ('Cutout', 0.8, 8)]
        image, boxes = translate_y_only_boxes(
            image, boxes, 0.2, translate_level_to_arg(2, TRANSLATE_BOX_CONST), REPLACE_VALUE)
        if should_apply_op(0.8):
            image = cutout(image, int((8 / MAX_LEVEL) * CUTOUT_CONST), REPLACE_VALUE)
    elif tf.equal(i, 2):  # [('Sharpness', 0.0, 8), ('ShearXBox', 0.4, 0)]
        if should_apply_op(0.):
            image = sharpness(image, enhance_level_to_arg(8))
        if should_apply_op(1.):
            image, boxes = shear_with_boxes(image, boxes, shrink_level_to_arg(2), REPLACE_VALUE, True)
    elif tf.equal(i, 3):  # [('ShearYBox', 1.0, 2), ('TranslateYOnlyBoxes', 0.6, 6)]
        if should_apply_op(1.):
            image, boxes = shear_with_boxes(image, boxes, shrink_level_to_arg(2), REPLACE_VALUE, False)
        image, boxes = translate_y_only_boxes(
            image, boxes, 0.6, translate_level_to_arg(6, TRANSLATE_BOX_CONST), REPLACE_VALUE)
    elif tf.equal(i, 4):  # [('RotateBox', 0.6, 10), ('Color', 1.0, 6)]
        if tf.less(tf.random.uniform([]), 0.6):
            image, boxes = rotate_with_boxes(image, boxes, rotate_level_to_arg(10), REPLACE_VALUE)
        if tf.less(tf.random.uniform([]), 1.):
            image = color(image, enhance_level_to_arg(6))
    elif tf.equal(i, 5):  # [('Color', 0.0, 0), ('ShearXOnlyBoxes', 0.8, 4)]
        if tf.less(tf.random.uniform([]), 0.):
            image = color(image, enhance_level_to_arg(0))
        image, boxes = shear_x_only_boxes(image, boxes, 0.8, shear_level_to_arg(4), REPLACE_VALUE)
    elif tf.equal(i, 6):  # [('ShearYOnlyBoxes', 0.8, 2), ('FlipOnlyBoxes', 0.0, 10)]
        image, boxes = shear_y_only_boxes(image, boxes, 0.8, shear_level_to_arg(2), REPLACE_VALUE)
        image, boxes = flip_only_boxes(image, boxes, 0.0)
    elif tf.equal(i, 7):  # [('Equalize', 0.6, 10), ('TranslateXBox', 0.2, 2)]
        if should_apply_op(0.6):
            image = equalize(image)
        if should_apply_op(0.2):
            image, boxes = translate_boxes(
                image, boxes, translate_level_to_arg(2, TRANSLATE_CONST), True, REPLACE_VALUE)
    elif tf.equal(i, 8):  # [('Color', 1.0, 10), ('TranslateYOnlyBoxes', 0.4, 6)]
        if should_apply_op(1.0):
            image = color(image, enhance_level_to_arg(10))
        image, boxes = translate_y_only_boxes(
            image, boxes, 0.4, translate_level_to_arg(6, TRANSLATE_BOX_CONST), REPLACE_VALUE)
    elif tf.equal(i, 9):  # [('RotateBox', 0.8, 10), ('Contrast', 0.0, 10)]
        if should_apply_op(0.8):
            image, boxes = rotate_with_boxes(image, boxes, rotate_level_to_arg(10), REPLACE_VALUE)
        if should_apply_op(0.0):
            image = contrast(image, enhance_level_to_arg(10))
    elif tf.equal(i, 10):  # [('Cutout', 0.2, 2), ('Brightness', 0.8, 10)]
        if should_apply_op(0.2):
            image = cutout(image, int((2 / MAX_LEVEL) * CUTOUT_CONST))
        if should_apply_op(0.8):
            image = brightness(image, enhance_level_to_arg(10))
    elif tf.equal(i, 11):  # [('Color', 1.0, 6), ('Equalize', 1.0, 2)]
        if should_apply_op(1.0):
            image = color(image, enhance_level_to_arg(6))
        if should_apply_op(1.0):
            image = equalize(image)
    elif tf.equal(i, 12):  # [('CutoutOnlyBoxes', 0.4, 6), ('TranslateYOnlyBoxes', 0.8, 2)]
        image, boxes = cutout_only_boxes(
            image, boxes, 0.4, int((6 / MAX_LEVEL) * CUTOUT_BOX_CONST), REPLACE_VALUE)
        image, boxes = translate_x_only_boxes(
            image, boxes, 0.8, translate_level_to_arg(2, TRANSLATE_BOX_CONST), REPLACE_VALUE)
    elif tf.equal(i, 13):  # [('Color', 0.2, 8), ('RotateBox', 0.8, 10)]
        if should_apply_op(0.2):
            image = color(image, enhance_level_to_arg(8))
        if should_apply_op(1.0):
            image = equalize(image)
    elif tf.equal(i, 14):  # [('Sharpness', 0.4, 4), ('TranslateYOnlyBoxes', 0.0, 4)]
        if should_apply_op(0.4):
            image = sharpness(image, enhance_level_to_arg(4))
        image, boxes = translate_y_only_boxes(
            image, boxes, 0.0, translate_level_to_arg(4, TRANSLATE_BOX_CONST), REPLACE_VALUE)
    elif tf.equal(i, 15):  # [('Sharpness', 1.0, 4), ('SolarizeAdd', 0.4, 4)]
        if should_apply_op(1.0):
            image = sharpness(image, enhance_level_to_arg(4))
        if should_apply_op(0.4):
            image = solarize_add(image, int((4 / MAX_LEVEL) * 110))
    elif tf.equal(i, 16):  # [('RotateBox', 1.0, 8), ('Sharpness', 0.2, 8)]
        if should_apply_op(1.0):
            image, boxes = rotate_with_boxes(image, boxes, rotate_level_to_arg(8), REPLACE_VALUE)
        if should_apply_op(0.2):
            image = sharpness(image, enhance_level_to_arg(8))
    elif tf.equal(i, 17):  # [('ShearYBox', 0.6, 10), ('EqualizeOnlyBoxes', 0.6, 8)]
        if should_apply_op(0.6):
            image, boxes = shear_with_boxes(image, boxes, shear_level_to_arg(10), REPLACE_VALUE, False)
        image, boxes = equalize_only_boxes(image, boxes, 0.6)
    elif tf.equal(i, 18):  # [('ShearXBox', 0.2, 6), ('TranslateYOnlyBoxes', 0.2, 10)]
        if should_apply_op(0.2):
            image, boxes = shear_with_boxes(image, boxes, shear_level_to_arg(6), REPLACE_VALUE, True)
        image, boxes = translate_y_only_boxes(
            image, boxes, 0.2, translate_level_to_arg(10, TRANSLATE_BOX_CONST), REPLACE_VALUE)
    else:  # [('SolarizeAdd', 0.6, 8), ('Brightness', 0.8, 10)]
        if should_apply_op(0.6):
            image = solarize_add(image, int((8 / MAX_LEVEL) * 110))
        if should_apply_op(0.8):
            image = brightness(image, enhance_level_to_arg(10))

    return image, boxes


def policy_v2(image, boxes):
    """Additional policy that performs well on object detection."""

    # Each tuple is an augmentation operation of the form
    # (operation, probability, magnitude). Each element in policy is a
    # sub-policy that will be applied sequentially on the image.

    i = tf.random.uniform([], 0, 15, tf.int32)

    if tf.equal(i, 0):  # [('Color', 0.0, 6), ('Cutout', 0.6, 8), ('Sharpness', 0.4, 8)]
        if should_apply_op(0.0):
            image = color(image, enhance_level_to_arg(6))
        if should_apply_op(0.6):
            image = cutout(image, int((8 / MAX_LEVEL) * CUTOUT_CONST), REPLACE_VALUE)
        if should_apply_op(0.4):
            image = sharpness(image, enhance_level_to_arg(8))
    elif tf.equal(i, 1):  # [('RotateBox', 0.4, 8), ('Sharpness', 0.4, 2), ('RotateBox', 0.8, 10)]
        if should_apply_op(0.4):
            image, boxes = rotate_with_boxes(image, boxes, rotate_level_to_arg(8), REPLACE_VALUE)
        if should_apply_op(0.4):
            image = sharpness(image, enhance_level_to_arg(2))
        if should_apply_op(0.8):
            image, boxes = rotate_with_boxes(image, boxes, rotate_level_to_arg(10), REPLACE_VALUE)
    elif tf.equal(i, 2):  # [('TranslateYBox', 1.0, 8), ('AutoContrast', 0.8, 2)]
        if should_apply_op(1.0):
            image, boxes = translate_boxes(image, boxes, rotate_level_to_arg(8), REPLACE_VALUE, False)
        if should_apply_op(0.8):
            image = autocontrast(image)
    elif tf.equal(i, 3):  # [('AutoContrast', 0.4, 6), ('ShearXBox', 0.8, 8), ('Brightness', 0.0, 10)]
        if should_apply_op(0.4):
            image = autocontrast(image)
        if should_apply_op(0.8):
            image, boxes = shear_with_boxes(image, boxes, shear_level_to_arg(8), REPLACE_VALUE, True)
        if should_apply_op(0.0):
            image = brightness(image, enhance_level_to_arg(10))
    elif tf.equal(i, 4):  # [('SolarizeAdd', 0.2, 6), ('Contrast', 0.0, 10), ('AutoContrast', 0.6, 0)]
        if should_apply_op(0.2):
            image = solarize_add(image, int((6 / MAX_LEVEL) * 110))
        if should_apply_op(0.0):
            image = contrast(image, enhance_level_to_arg(10))
        if should_apply_op(0.6):
            image = autocontrast(image)
    elif tf.equal(i, 5):  # [('Cutout', 0.2, 0), ('Solarize', 0.8, 8), ('Color', 1.0, 4)]
        if should_apply_op(0.2):
            image = cutout(image, int((0 / MAX_LEVEL) * CUTOUT_CONST), REPLACE_VALUE)
        if should_apply_op(0.8):
            image = solarize(image)
        if should_apply_op(1.0):
            image = color(image, enhance_level_to_arg(4))
    elif tf.equal(i, 6):  # [('TranslateYBox', 0.0, 4), ('Equalize', 0.6, 8), ('Solarize', 0.0, 10)]
        if should_apply_op(0.0):
            image, boxes = translate_boxes(
                image, boxes, translate_level_to_arg(4, TRANSLATE_CONST), REPLACE_VALUE, False)
        if should_apply_op(0.6):
            image = equalize(image)
        if should_apply_op(0.0):
            image = solarize(image)
    elif tf.equal(i, 7):  # [('TranslateYBox', 0.2, 2), ('ShearYBox', 0.8, 8), ('RotateBox', 0.8, 8)]
        if should_apply_op(0.2):
            image, boxes = translate_boxes(
                image, boxes, translate_level_to_arg(2, TRANSLATE_CONST), REPLACE_VALUE, False)
        if should_apply_op(0.8):
            image, boxes = shear_with_boxes(image, boxes, shear_level_to_arg(8), REPLACE_VALUE, False)
        if should_apply_op(0.8):
            image, boxes = rotate_with_boxes(image, boxes, rotate_level_to_arg(8), REPLACE_VALUE)
    elif tf.equal(i, 8):  # [('Cutout', 0.8, 8), ('Brightness', 0.8, 8), ('Cutout', 0.2, 2)]
        if should_apply_op(0.8):
            image = cutout(image, int((8 / MAX_LEVEL) * CUTOUT_CONST), REPLACE_VALUE)
        if should_apply_op(0.8):
            image = brightness(image, enhance_level_to_arg(8))
        if should_apply_op(0.2):
            image = cutout(image, int((2 / MAX_LEVEL) * CUTOUT_CONST), REPLACE_VALUE)
    elif tf.equal(i, 9):  # [('Color', 0.8, 4), ('TranslateYBox', 1.0, 6), ('RotateBox', 0.6, 6)]
        if should_apply_op(0.8):
            image = color(image, enhance_level_to_arg(4))
        if should_apply_op(1.0):
            image, boxes = translate_boxes(
                image, boxes, translate_level_to_arg(6, TRANSLATE_CONST), REPLACE_VALUE, False)
        if should_apply_op(0.6):
            image, boxes = rotate_with_boxes(image, boxes, rotate_level_to_arg(6), REPLACE_VALUE)
    elif tf.equal(i, 10):  # [('RotateBox', 0.6, 10), ('BoxCutout', 1.0, 4), ('Cutout', 0.2, 8)]
        if should_apply_op(0.6):
            image, boxes = rotate_with_boxes(image, boxes, rotate_level_to_arg(10), REPLACE_VALUE)
        if should_apply_op(1.0):
            image, boxes = box_cutout(
                 image, boxes, *bbox_cutout_level_to_arg(4, CUTOUT_MAX_PAD_FRACTION, CUTOUT_BOX_REPLACE_WITH_MEAN))
        if should_apply_op(0.2):
            image = cutout(image, int((8 / MAX_LEVEL) * CUTOUT_CONST), REPLACE_VALUE)
    elif tf.equal(i, 11):  # [('RotateBox', 0.0, 0), ('Equalize', 0.6, 6), ('ShearYBox', 0.6, 8)]
        if should_apply_op(0.0):
            image, boxes = rotate_with_boxes(image, boxes, rotate_level_to_arg(0), REPLACE_VALUE)
        if should_apply_op(0.6):
            image = equalize(image)
        if should_apply_op(0.6):
            image, boxes = shear_with_boxes(image, boxes, shear_level_to_arg(8), REPLACE_VALUE, False)
    elif tf.equal(i, 12):  # [('Brightness', 0.8, 8), ('AutoContrast', 0.4, 2), ('Brightness', 0.2, 2)]
        if should_apply_op(0.8):
            image = brightness(image, enhance_level_to_arg(8))
        if should_apply_op(0.4):
            image = autocontrast(image)
        if should_apply_op(0.2):
            image = brightness(image, enhance_level_to_arg(2))
    elif tf.equal(i, 13):  # [('TranslateYBox', 0.4, 8), ('Solarize', 0.4, 6), ('SolarizeAdd', 0.2, 10)]
        if should_apply_op(0.4):
            image, boxes = translate_boxes(
                image, boxes, translate_level_to_arg(8, TRANSLATE_CONST), REPLACE_VALUE, False)
        if should_apply_op(0.4):
            image = solarize(image)
        if should_apply_op(0.2):
            image = solarize_add(image, int((10 / MAX_LEVEL) * 110))
    else:  # [('Contrast', 1.0, 10), ('SolarizeAdd', 0.2, 8), ('Equalize', 0.2, 4)],
        if should_apply_op(1.0):
            image = contrast(image, enhance_level_to_arg(10))
        if should_apply_op(0.2):
            image = solarize_add(image, int((8 / MAX_LEVEL) * 110))
        if should_apply_op(0.2):
            image = equalize(image)

    return image, boxes


@tf.function
def policy_v3(image, boxes):
    """"Additional policy that performs well on object detection."""
    # Each tuple is an augmentation operation of the form
    # (operation, probability, magnitude). Each element in policy is a
    # sub-policy that will be applied sequentially on the image.
    i = tf.random.uniform([], 0, 15, tf.int32)

    if tf.equal(i, 0):  # [('Posterize', 0.8, 2), ('TranslateXBox', 1.0, 8)]
        if should_apply_op(0.8):
            image = posterize(image, int((2 / MAX_LEVEL) * 4))
        if should_apply_op(1.0):
            image, boxes = translate_boxes(
                image, boxes, translate_level_to_arg(8, TRANSLATE_CONST), REPLACE_VALUE, True)
    elif tf.equal(i, 1):  # [('BoxCutout', 0.2, 10), ('Sharpness', 1.0, 8)]
        if should_apply_op(0.2):
            image, boxes = box_cutout(
                image, boxes, *bbox_cutout_level_to_arg(10, CUTOUT_MAX_PAD_FRACTION, CUTOUT_BOX_REPLACE_WITH_MEAN))
        if should_apply_op(1.0):
            image = sharpness(image, enhance_level_to_arg(8))
    elif tf.equal(i, 2):  # [('RotateBox', 0.6, 8), ('RotateBox', 0.8, 10)]
        if should_apply_op(0.6):
            image, boxes = rotate_with_boxes(image, boxes, rotate_level_to_arg(8), REPLACE_VALUE)
        if should_apply_op(0.8):
            image, boxes = rotate_with_boxes(image, boxes, rotate_level_to_arg(10), REPLACE_VALUE)
    elif tf.equal(i, 3):  # [('Equalize', 0.8, 10), ('AutoContrast', 0.2, 10)]
        if should_apply_op(0.8):
            image = equalize(image)
        if should_apply_op(0.2):
            image = autocontrast(image)
    elif tf.equal(i, 4):   # [('SolarizeAdd', 0.2, 2), ('TranslateYBox', 0.2, 8)]
        if should_apply_op(0.2):
            image = solarize_add(image, int((2 / MAX_LEVEL) * 110))
        if should_apply_op(0.2):
            image, boxes = translate_boxes(
                image, boxes, translate_level_to_arg(8, TRANSLATE_CONST), REPLACE_VALUE, False)
    elif tf.equal(i, 5):  # [('Sharpness', 0.0, 2), ('Color', 0.4, 8)]
        if should_apply_op(0.0):
            image = sharpness(image, enhance_level_to_arg(2))
        if should_apply_op(0.4):
            image = color(image, enhance_level_to_arg(8))
    elif tf.equal(i, 6):  # [('Equalize', 1.0, 8), ('TranslateYBox', 1.0, 8)]
        if should_apply_op(1.0):
            image = equalize(image)
        if should_apply_op(1.0):
            image, boxes = translate_boxes(
                image, boxes, translate_level_to_arg(8, TRANSLATE_CONST), REPLACE_VALUE, False)
    elif tf.equal(i, 7):  # [('Posterize', 0.6, 2), ('RotateBox', 0.0, 10)]
        if should_apply_op(0.6):
            image = posterize(image, int((2 / MAX_LEVEL) * 4))
        if should_apply_op(0.0):
            image, boxes = rotate_with_boxes(image, boxes, rotate_level_to_arg(10), REPLACE_VALUE)
    elif tf.equal(i, 8):  # [('AutoContrast', 0.6, 0), ('RotateBox', 1.0, 6)]
        if should_apply_op(0.6):
            image = autocontrast(image)
        if should_apply_op(1.0):
            image, boxes = rotate_with_boxes(image, boxes, rotate_level_to_arg(6), REPLACE_VALUE)
    elif tf.equal(i, 9):  # [('Equalize', 0.0, 4), ('Cutout', 0.8, 10)]
        if should_apply_op(0.0):
            image = equalize(image)
        if should_apply_op(0.8):
            image = cutout(image, int((10 / MAX_LEVEL) * CUTOUT_CONST), REPLACE_VALUE)
    elif tf.equal(i, 10):  # [('Brightness', 1.0, 2), ('TranslateYBox', 1.0, 6)]
        if should_apply_op(1.0):
            image = brightness(image, enhance_level_to_arg(2))
        if should_apply_op(1.0):
            image, boxes = translate_boxes(
                image, boxes, translate_level_to_arg(6, TRANSLATE_CONST), REPLACE_VALUE, False)
    elif tf.equal(i, 11):  # [('Contrast', 0.0, 2), ('ShearYBox', 0.8, 0)]
        if should_apply_op(0.0):
            image = contrast(image, enhance_level_to_arg(2))
        if should_apply_op(0.8):
            image, boxes = shear_with_boxes(image, boxes, shear_level_to_arg(0), REPLACE_VALUE, False)
    elif tf.equal(i, 12):  # [('AutoContrast', 0.8, 10), ('Contrast', 0.2, 10)]
        if should_apply_op(0.8):
            image = autocontrast(image)
        if should_apply_op(0.2):
            image = contrast(image, enhance_level_to_arg(10))
    elif tf.equal(i, 13):  # [('RotateBox', 1.0, 10), ('Cutout', 1.0, 10)]
        if should_apply_op(1.0):
            image, boxes = rotate_with_boxes(image, boxes, rotate_level_to_arg(10), REPLACE_VALUE)
        if should_apply_op(1.0):
            image = cutout(image, int((10 / MAX_LEVEL) * CUTOUT_CONST), REPLACE_VALUE)
    else:  # [('SolarizeAdd', 0.8, 6), ('Equalize', 0.8, 8)]
        if should_apply_op(0.8):
            image = solarize_add(image, int((6 / MAX_LEVEL) * 110))
        if should_apply_op(0.8):
            image = equalize(image)

    return image, boxes


def policy(version="v0"):
    if version == "v0":
        return policy_v0
    elif version == "v1":
        return policy_v1
    elif version == "v2":
        return policy_v2
    elif version == "v3":
        return policy_v3
    else:
        return policy_v0


def test():
    import cv2
    import numpy as np

    image = cv2.imread("/Users/bailang/Desktop/Workspace/face_detections/data/0_Parade_marchingband_1_353.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_size = image.shape
    boxes = np.array([[263, 381, 113, 169],
                      [635, 271, 134, 169]])
    for l in boxes:
        pt1 = (int(l[0]), int(l[1]))
        pt2 = (int(l[0] + l[2]), int(l[1] + l[3]))
        print(pt1, pt2)
        image = cv2.rectangle(image, pt1=pt1, pt2=pt2, thickness=1, color=(255, 0, 0))

    boxes = np.stack([boxes[:, 1], boxes[:, 0], boxes[:, 1] + boxes[:, 3], boxes[:, 0] + boxes[:, 2]], axis=1)
    boxes = boxes / np.array([image_size[0], image_size[1], image_size[0], image_size[1]])

    image_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)
    boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
    image_tensor, boxes = policy_v2(image_tensor, boxes)
    img = image_tensor.numpy()
    img = img.astype(np.uint8)
    img_size = img.shape
    boxes = boxes.numpy()
    for l in boxes:
        pt1 = (int(l[1] * img_size[1]), int(l[0] * img_size[0]))
        pt2 = (int(l[3] * img_size[1]), int(l[2] * img_size[0]))
        print(pt1, pt2)
        img = cv2.rectangle(img, pt1=pt1, pt2=pt2, thickness=1, color=(0, 255, 0))

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("image", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    test()


