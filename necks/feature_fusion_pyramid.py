import tensorflow as tf
from core.layers import conv_block


def feature_fusion_pyramid(inputs,
                           convolution="conv2d",
                           normalization="batch_norm",
                           activation="relu",
                           output_filters=(),
                           num_outputs=6,
                           group=32,
                           weight_decay=0.,
                           add_extra_conv=False,
                           use_multiplication=False):
    assert len(inputs) == len(output_filters)
    num_inputs = len(inputs)

    output_filters = [output_filters] * num_inputs\
        if isinstance(output_filters, (int, float)) else output_filters

    # build top-down path
    kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    for i in range(num_inputs - 1, 0, -1):
        top = tf.keras.layers.Conv2DTranspose(filters=output_filters[i-1],
                                              kernel_size=(4, 4),
                                              strides=(2, 2),
                                              padding="same",
                                              kernel_regularizer=kernel_regularizer)(inputs[i])
        if use_multiplication:
            inputs[i-1] = tf.keras.layers.Multiply()([top, inputs[i-1]])
        else:
            inputs[i-1] = tf.keras.layers.Add()([top, inputs[i-1]])
        inputs[i-1] = conv_block(convolution,
                                 filters=256,
                                 kernel_size=(1, 1),
                                 strides=(1, 1),
                                 kernel_regularizer=kernel_regularizer,
                                 normalization=normalization,
                                 group=group,
                                 activation=activation,
                                 name="reduced_conv2d_" + str(i+1))(inputs[i-1])

    inputs[-1] = conv_block(convolution,
                            filters=256,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            kernel_regularizer=kernel_regularizer,
                            normalization=normalization,
                            group=group,
                            activation=activation,
                            name="reduced_conv2d_" + str(i + 1))(inputs[-1])

    for i in range(num_inputs, num_outputs):
        if add_extra_conv:
            inputs.append(conv_block(convolution,
                                     filters=256,
                                     kernel_size=(3, 3),
                                     strides=(2, 2),
                                     kernel_regularizer=kernel_regularizer,
                                     normalization=normalization,
                                     group=group,
                                     activation=activation,
                                     name="reduced_conv2d_" + str(i + 1)))
        else:
            inputs.append(tf.keras.layers.MaxPool2D(
                (2, 2), (2, 2), "same", name="extra_max_pool_" + str(i+1))(inputs[-1]))

    return inputs

