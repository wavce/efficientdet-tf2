import tensorflow as tf
from core.layers import conv_block
from configs.params_dict import ParamsDict
from core.layers import build_normalization

def fpn(inputs,
        convolution="separable_conv2d",
        normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True),
        activation=dict(activation="relu"),
        feat_dims=64,
        min_level=3,
        max_level=7, 
        use_multiplication=False,
        dropblock=None,
        weight_decay=0.,
        add_extra_conv=False,
        name="fpn_neck",
        **Kwargs):
    activation = activation.as_dict() if isinstance(activation, ParamsDict) else activation
    laterals = []
    num_outputs = max_level - min_level + 1
    num_inputs = len(inputs)
    kernel_regularizer = (tf.keras.regularizers.l2(weight_decay) 
                          if weight_decay is not None and weight_decay > 0 else None) 
  
    for i, feat in enumerate(inputs):
        laterals.append(conv_block(convolution="conv2d",
                                   filters=feat_dims,
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   kernel_regularizer=kernel_regularizer,
                                   normalization=normalization,
                                   activation=activation,
                                   dropblock=dropblock,
                                   name=name+"/lateral_%s_%d" % (convolution, i))(feat))

    for i in range(num_inputs - 1, 0, -1):
        top = tf.keras.layers.UpSampling2D(size=(2, 2), name=name + "/upsample_%d" % i)(laterals[i])
        if use_multiplication:
            laterals[i-1] = tf.keras.layers.Multiply(name=name+"/multiply_"+str(i))([laterals[i-1], top])
        else:
            laterals[i-1] = tf.keras.layers.Add(name=name+"/sum_"+str(i))([laterals[i-1], top])

    # Adds post-hoc 3x3 convolution kernel.
    for i in range(num_inputs):
        laterals[i] = conv_block(convolution=convolution,
                                 filters=feat_dims,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 kernel_regularizer=kernel_regularizer,
                                 normalization=normalization,
                                 activation=activation,
                                 dropblock=dropblock,
                                 name=name+"/post_hoc_%s_%d" % (convolution, i))(laterals[i])

    for i in range(num_inputs, num_outputs):
        if add_extra_conv:
            laterals.append(conv_block(convolution=convolution,
                                       filters=feat_dims,
                                       kernel_size=(3, 3),
                                       strides=(2, 2),
                                       kernel_regularizer=kernel_regularizer,
                                       normalization=normalization,
                                       activation=activation,
                                       dropblock=dropblock,
                                       name=name+"/post_hoc_conv2d_" + str(i))(laterals[-1]))
        else:
            laterals.append(tf.keras.layers.MaxPool2D(
                (2, 2), (2, 2), "same", name=name + "/max_pool2d_" + str(i))(laterals[-1]))

    return laterals

