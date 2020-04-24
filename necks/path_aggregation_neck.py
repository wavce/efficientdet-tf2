import tensorflow as tf
from core.layers import conv_block


def path_aggregation_neck(inputs,
                          convolution="conv2d",
                          normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True),
                          activation=dict(activation="relu"),
                          feat_dims=64,
                          min_level=3,
                          max_level=7,
                          add_extra_conv=False,
                          dropblock=None,
                          weight_decay=0.,
                          use_multiplication=False,
                          name="path_aggregation_neck"):
    kernel_regularizer = (tf.keras.regularizers.l2(weight_decay) 
                          if weight_decay is not None and weight_decay > 0 else None) 
    num_outputs = max_level - min_level + 1
    output_filters = [output_filters] * num_outputs \
        if isinstance(output_filters, int) else output_filters
    features = []
    num_inputs = len(inputs)
    for i, features in enumerate(inputs):
        x = conv_block(convolution="conv2d",
                       filters=feat_dims,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       kernel_regularizer=kernel_regularizer,
                       normalization=normalization,
                       activation=activation,
                       dropblock=dropblock,
                       name="top_down_conv2d_%d" % (i+1))(features)
        features.append(x)

    for i in range(num_inputs - 1, 0, -1):
        top = tf.keras.layers.UpSampling2D((2, 2), interpolation="nearest")(features[i+1])
        if use_multiplication:
            features[i] = tf.keras.layers.Multiply()([features[i], top])
        else:
            features[i] = tf.keras.layers.Add()([features[i], top])

    for i in range(1, num_inputs):
        x = conv_block(convolution="conv2d",
                       filters=feat_dims,
                       kernel_size=(3, 3),
                       strides=(2, 2),
                       kernel_regularizer=kernel_regularizer,
                       normalization=normalization,
                       activation=activation,
                       dropblock=dropblock,
                       name="bottom_up_conv2d_%d" % (i+1))(features[i-1])
        if use_multiplication:
            features[i] = tf.keras.layers.Multiply()([x, features[i]])
        else:
            features[i] = tf.keras.layers.Add()([x, features[i]])

    for i in range(num_inputs, num_outputs):
        if add_extra_conv:
            features.append((conv_block(convolution,
                                        filters=output_filters[i],
                                        kernel_size=(3, 3),
                                        strides=(2, 2),
                                        kernel_regularizer=kernel_regularizer,
                                        normalization=normalization,
                                        group=group,
                                        activation=activation,
                                        name="extra_conv2d_%d" % (i + 1))(features[-1])))
        else:
            features.append(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                                      strides=(2, 2))(features[-1]))

    return features
