import tensorflow as tf
from .activations import Swish
from .max_in_out import MaxInOut
from .drop_block import DropBlock2D
from .nms import FastNonMaxSuppression
from .nms import BatchNonMaxSuppression
from .nms import CombinedNonMaxSuppression
from .nms import BatchSoftNonMaxSuppression
from .normalizations import L2Normalization
from .normalizations import GroupNormalization
from .deformable_conv2d import DeformableConv2D
from .nearest_upsamling import NearestUpsampling2D
from .weight_standardization_conv2d import WSConv2D
from .normalizations import SwitchableNormalization
from .normalizations import FilterResponseNormalization
from .position_sensitive_roi_pooling import PSRoIPooling
from .position_sensitive_average_pooling import PSAvgPooling


def build_convolution(convolution, **kwargs):
    if convolution == "depthwise_conv2d":
        return tf.keras.layers.DepthwiseConv2D(**kwargs)
    elif convolution == "wsconv2d":
        return WSConv2D(**kwargs)
    elif convolution == "conv2d":
        return tf.keras.layers.Conv2D(**kwargs)
    elif convolution == "separable_conv2d":
        return tf.keras.layers.SeparableConv2D(**kwargs)
    elif convolution == "deformable_conv2d":
        return DeformableConv2D(**kwargs)
    else:
        raise TypeError("Could not interpret convolution function identifier: {}".format(repr(convolution)))


def build_normalization(**kwargs):
    normalization = kwargs.pop("normalization")
    if normalization == "group_norm":
        return GroupNormalization(**kwargs)
    elif normalization == "batch_norm":
        return tf.keras.layers.BatchNormalization(**kwargs)
    elif normalization == "switchable_norm":
        return SwitchableNormalization(**kwargs)
    elif normalization == "filter_response_norm":
        return FilterResponseNormalization(**kwargs)
    else:
        raise TypeError("Could not interpret normalization function identifier: {}".format(
            repr(normalization)))


def build_activation(**kwargs):
    activation = kwargs.pop("activation")
    # if "swish" == activation:
    #     return Swish(**kwargs)
    return tf.keras.layers.Activation(activation, **kwargs)


def conv_block(convolution,
               filters,
               kernel_size,
               strides,
               kernel_regularizer,
               normalization,
               activation,
               dropblock=None,
               name=None):
    block = tf.keras.Sequential([build_convolution(convolution,
                                                   filters=filters,
                                                   kernel_size=kernel_size,
                                                   strides=strides,
                                                   padding="same",
                                                   use_bias=normalization is None,
                                                   kernel_regularizer=kernel_regularizer)],
                                name=name)
    if normalization is not None:
        block.add(build_normalization(**normalization))

    if activation is not None:
        build_activation(**activation)

    if dropblock:
        block.add(DropBlock2D(**dropblock))

    return block


NMS = {
    "batch_non_max_suppression": BatchNonMaxSuppression,
    "fast_non_max_suppression": FastNonMaxSuppression,
    "batch_soft_non_max_suppression": BatchSoftNonMaxSuppression,
    "combined_non_max_suppression": CombinedNonMaxSuppression
}

def build_nms(nms, **kwargs):
    return NMS[nms](**kwargs)


__all__ = [
    "Swish",
    "WSConv2D",
    "MaxInOut",
    "conv_block",
    "DropBlock2D",
    "PSAvgPooling",
    "L2Normalization",
    "build_activation",
    "build_convolution",
    "GroupNormalization",
    "build_normalization",
    "NearestUpsampling2D",
    "SwitchableNormalization",
    "BatchNonMaxSuppression",
    "CombinedNonMaxSuppression",
    "BatchSoftNonMaxSuppression",
    "FastNonMaxSuppression",
    "build_nms"
]
