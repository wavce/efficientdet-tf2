import numpy as np
import tensorflow as tf
from core.layers import DropBlock2D
from core.layers import build_convolution
from core.layers import build_normalization


def bottleneck_v1(x,
                  convolution,
                  filters,
                  strides=1,
                  dilation_rate=1,
                  normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                  activation=dict(activation="relu"),
                  trainable=True,
                  weight_decay=0.,
                  dropblock=None,
                  use_conv_shortcut=True,
                  name=None):
    """A residual block.

        Args:
            x: input tensor.
            filters: integer, filters of the bottleneck layer.
            convolution: The convolution type.
            strides: default 1, stride of the first layer.
            dilation_rate: default 1, dilation rate in 3x3 convolution.
            activation: the activation layer name.
            trainable: does this block is trainable.
            normalization: the normalization, e.g. "batch_norm", "group_norm" etc.
            weight_decay: weight decay.
            dropblock: the arguments in DropBlock2D
            use_conv_shortcut: default True, use convolution shortcut if True,
                otherwise identity shortcut.
            name: string, block label.
        Returns:
            Output tensor for the residual block.
    """
    if use_conv_shortcut is True:
        shortcut = build_convolution(convolution,
                                     filters=4 * filters,
                                     kernel_size=1,
                                     strides=strides,
                                     trainable=trainable,
                                     kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     name=name + "_0_conv")(x)
        shortcut = build_normalization(**normalization, name=name+"_0_bn")(shortcut)
    else:
        shortcut = x

    if dropblock is not None:
        shortcut = DropBlock2D(**dropblock, name=name + "_0_dropblock")(shortcut)

    x = build_convolution(convolution,
                          filters=filters,
                          kernel_size=1,
                          strides=strides,
                          trainable=trainable,
                          kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                          name=name + "_1_conv")(x)
    x = build_normalization(**normalization, name=name + "_1_bn")(x)
    x = tf.keras.layers.Activation(**activation, name=name + "_1_relu")(x)
    if dropblock is not None:
        x = DropBlock2D(**dropblock, name=name + "_1_dropblock")(x)

    x = build_convolution(convolution,
                          filters=filters,
                          kernel_size=3,
                          strides=1,
                          padding="SAME",
                          dilation_rate=dilation_rate,
                          trainable=trainable,
                          kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                          name=name + "_2_conv")(x)
    x = build_normalization(**normalization, name=name + "_2_bn")(x)
    x = tf.keras.layers.Activation(**activation, name=name + "_2_relu")(x)
    if dropblock is not None:
        x = DropBlock2D(**dropblock , name=name + "_2_dropblock")(x)

    x = build_convolution(convolution,
                          filters=4 * filters,
                          kernel_size=1,
                          trainable=trainable,
                          kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                          name=name + "_3_conv")(x)
    x = build_normalization(**normalization, name=name + "_3_bn")(x)
    if dropblock is not None:
        x = DropBlock2D(**dropblock, name=name + "_3_dropblock")(x)
    x = tf.keras.layers.Add(name=name + "_add")([shortcut, x])
    x = tf.keras.layers.Activation(**activation, name=name + "_out")(x)

    return x


def bottleneck_v2(x,
                  convolution,
                  filters,
                  strides=1,
                  dilation_rate=1,
                  normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                  activation=dict(activation="relu"),
                  trainable=True,
                  weight_decay=0.,
                  dropblock=None,
                  use_conv_shortcut=True,
                  name=None):
    """A residual block.

        Args:
            x: input tensor.
            filters: integer, filters of the bottleneck layer.
            convolution: The convolution type.
            strides: default 1, stride of the first layer.
            dilation_rate: default 1, dilation rate in 3x3 convolution.
            activation: the activation layer name.
            trainable: does this block is trainable.
            normalization: the normalization, e.g. "batch_norm", "group_norm" etc.
            weight_decay: weight decay.
            dropblock: the arguments in DropBlock2D.
            use_conv_shortcut: default True, use convolution shortcut if True,
                otherwise identity shortcut.
            name: string, block label.
    Returns:
        Output tensor for the residual block.
    """
    bn_axis = 3 if tf.keras.backend.image_data_format() == "channels_last" else 1

    preact = build_normalization(**normalization, name=name + "_preact_bn")(x)
    preact = tf.keras.layers.Activation(**activation, name=name + "_preact_relu")(preact)

    if use_conv_shortcut is True:
        shortcut = build_convolution(convolution,
                                     filters=4 * filters,
                                     kernel_size=1,
                                     strides=strides,
                                     trainable=trainable,
                                     kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     name=name + "_0_conv")(preact)
    else:
        shortcut = tf.keras.layers.MaxPooling2D(1, strides=strides)(x) if strides > 1 else x

    if dropblock is not None:
        shortcut = DropBlock2D(**dropblock, name=name + "_0_dropblock")(shortcut)

    x = build_convolution(convolution,
                          filters=filters,
                          kernel_size=1,
                          strides=1,
                          use_bias=False,
                          trainable=trainable,
                          kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                          name=name + "_1_conv")(preact)

    x = build_normalization(**normalization, name=name + "_1_bn")(x)
    x = tf.keras.layers.Activation(**activation, name=name + "_1_relu")(x)
    if dropblock is not None:
        x = DropBlock2D(**dropblock, name=name + "_1_dropblock")(x)

    x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + "_2_pad")(x)
    x = build_convolution(convolution,
                          filters=filters,
                          kernel_size=3,
                          strides=strides,
                          dilation_rate=1 if strides > 1 else dilation_rate,
                          use_bias=False,
                          trainable=trainable,
                          kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                          name=name + "_2_conv")(x)
    x = build_normalization(**normalization, name=name + "_2_bn")(x)
    x = tf.keras.layers.Activation(**activation, name=name + "_2_relu")(x)
    if dropblock is not None:
        x = DropBlock2D(**dropblock)(x)

    x = build_convolution(convolution,
                          filters=4 * filters,
                          kernel_size=1,
                          trainable=trainable,
                          kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                          name=name + "_3_conv")(x)
    if dropblock is not None:
        x = DropBlock2D(**dropblock, name=name + "_2_dropblock")(x)
    x = tf.keras.layers.Add(name=name + "_out")([shortcut, x])

    return x, preact


def bottleneckx(x,
                convolution,
                filters,
                strides=1,
                dilation_rate=1,
                cardinality=32,
                normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                activation=dict(activation="relu"),
                trainable=True,
                weight_decay=0.,
                use_conv_shortcut=True,
                name=None):
    """A residual block.
        Args:
            x: input tensor.
            filters: integer, filters of the bottleneck layer.
            convolution: The convolution type.
            strides: default 1, stride of the first layer.
            dilation_rate: default 1, dilation rate in 3x3 convolution.
            cardinality: default 32, the cardinality in resnext.
            activation: the activation layer name.
            trainable: does this block is trainable.
            normalization: the normalization, e.g. "batch_norm", "group_norm" etc.
            weight_decay: the weight decay.
            use_conv_shortcut: default True, use convolution shortcut if True,
                otherwise identity shortcut.
            name: string, block label.
        Returns:
            Output tensor for the residual block.
    """
    bn_axis = 3 if tf.keras.backend.image_data_format() == "channels_last" else 1

    if use_conv_shortcut:
        shortcut = build_convolution(convolution,
                                     filters=(64 // cardinality) * filters,
                                     kernel_size=1,
                                     strides=strides,
                                     use_bias=False,
                                     trainable=trainable,
                                     kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     name=name + "_0_conv")(x)
        shortcut = build_normalization(**normalization, name=name + "_0_bn")(shortcut)
    else:
        shortcut = x

    x = tf.keras.layers.Conv2D(filters, 1, use_bias=False, trainable=trainable, name=name + "_1_conv")(x)
    x = build_normalization(**normalization, name=name + "_1_bn")(x)
    x = tf.keras.layers.Activation(activation, name=name + "_1_relu")(x)

    c = filters // cardinality
    x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + "_2_pad")(x)
    x = tf.keras.layers.DepthwiseConv2D(
        3, strides=strides, depth_multiplier=c, dilation_rate=dilation_rate,
        use_bias=False, trainable=trainable, name=name + "_2_conv")(x)
    kernel = np.zeros((1, 1, filters * c, filters), dtype=np.float32)
    for i in range(filters):
        start = (i // c) * c * c + i % c
        end = start + c * c
        kernel[:, :, start:end:c, i] = 1.

    x = tf.keras.layers.Conv2D(
        filters, 1, use_bias=False, trainable=False,
        kernel_initializer={
            "class_name": "Constant",
            "config": {"value": kernel}
        },
        name=name + "_2_gconv")(x)
    x = build_normalization(**normalization, name=name + "_2_bn")(x)
    x = tf.keras.layers.Activation("relu", name=name + "_2_relu")(x)

    x = tf.keras.layers.Conv2D(
        (64 // cardinality) * filters, 1, use_bias=False, trainable=trainable, name=name + "_3_conv")(x)
    x = build_normalization(**normalization, name=name + "_3_bn")(x)

    x = tf.keras.layers.Add(name=name + "_add")([shortcut, x])
    x = tf.keras.layers.Activation("relu", name=name + "_out")(x)

    return x


class Subsample(tf.keras.layers.Layer):
    def __init__(self, factor, **kwargs):
        super(Subsample, self).__init__(**kwargs)
        self.pool = None
        if factor != 1:
            self.pool = tf.keras.layers.MaxPool2D((1, 1), factor)

    def call(self, inputs):
        if self.pool is not None:
            return self.pool(inputs)

        return inputs
