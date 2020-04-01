import os
import copy
import math
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.applications import imagenet_utils
from collections import namedtuple
from backbones.backbone import Backbone
from core.layers import build_activation
from core.layers import build_normalization


DEFAULT_BLOCKS_ARGS = [{
    'kernel_size': 3,
    'repeats': 1,
    'filters_in': 32,
    'filters_out': 16,
    'expand_ratio': 1,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 2,
    'filters_in': 16,
    'filters_out': 24,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 2,
    'filters_in': 24,
    'filters_out': 40,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 3,
    'filters_in': 40,
    'filters_out': 80,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 3,
    'filters_in': 80,
    'filters_out': 112,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 4,
    'filters_in': 112,
    'filters_out': 192,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 1,
    'filters_in': 192,
    'filters_out': 320,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


class EfficientNet(Backbone):
    PARAMS = {
            # (width_coefficient, depth_coefficient, resolution, dropout_rate)
            "efficientnetb0": (1.0, 1.0, 224, 0.2),
            "efficientnetb1": (1.0, 1.1, 240, 0.2),
            "efficientnetb2": (1.1, 1.2, 260, 0.3),
            "efficientnetb3": (1.2, 1.4, 300, 0.3),
            "efficientnetb4": (1.4, 1.8, 380, 0.4),
            "efficientnetb5": (1.6, 2.2, 456, 0.4),
            "efficientnetb6": (1.8, 2.6, 528, 0.5),
            "efficientnetb7": (2.0, 3.1, 600, 0.5),
        }

    def __init__(self,
                 name,
                 convolution='conv2d', 
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True), 
                 activation={"activation": "swish"}, 
                 output_indices=(3, 4), 
                 strides=(2, 2, 2, 2, 2), 
                 dilation_rates=(1,1,1,1,1), 
                 frozen_stages=( -1), 
                 weight_decay=0.0, 
                 dropblock=None, 
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 blocks_args="default",
                 input_shape=None,
                 input_tensor=None,
                 classifier_activation='softmax',
                 pretrained_weights_path=None):
        
        self.bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
        self.width_coefficient = EfficientNet.PARAMS[name][0]
        self.depth_coefficient = EfficientNet.PARAMS[name][1]
        default_size = EfficientNet.PARAMS[name][2]
        dropout_rate = EfficientNet.PARAMS[name][3]
        default_shape = [default_size, default_size, 3] if self.bn_axis == 3 else [3, default_size, default_size]
        input_shape = input_shape or default_shape

        super().__init__(name, 
                         convolution=convolution,
                         normalization=normalization, 
                         activation=activation, 
                         output_indices=output_indices, 
                         strides=strides, 
                         dilation_rates=dilation_rates, 
                         frozen_stages=frozen_stages, 
                         weight_decay=weight_decay, 
                         dropblock=dropblock, 
                         input_shape=input_shape,
                         input_tensor=input_tensor,
                         pretrained_weights_path=pretrained_weights_path)
        """Instantiates the EfficientNet architecture using given scaling coefficients.

            Optionally loads weights pre-trained on ImageNet.
            Note that the data format convention used by the model is
            the one specified in your Keras config at `~/.keras/keras.json`.

            Arguments:
                width_coefficient: float, scaling coefficient for network width.
                depth_coefficient: float, scaling coefficient for network depth.
                default_size: integer, default input image size.
                dropout_rate: float, dropout rate before final classifier layer.
                drop_connect_rate: float, dropout rate at skip connections.
                depth_divisor: integer, a unit of network width.
                blocks_args: list of dicts, parameters to construct block modules.
                input_tensor: optional Keras tensor
                    (i.e. output of `layers.Input()`)
                    to use as image input for the model.
                input_shape: optional shape tuple, only to be specified
                    if `include_top` is False.
                    It should have exactly 3 inputs channels.
                classes: optional number of classes to classify images
                    into, only to be specified if `include_top` is True, and
                    if no `weights` argument is specified.
        """
        if blocks_args == 'default':
            blocks_args = DEFAULT_BLOCKS_ARGS
        
        self.blocks_args = blocks_args
        self.depth_divisor = depth_divisor
        self.drop_connect_rate = drop_connect_rate
        self.dropout_rate = dropout_rate
        self.classifier_activation = classifier_activation

    def round_filters(self, filters):
        """Round number of filters based on depth multiplier."""
        filters *= self.width_coefficient
        new_filters = max(self.depth_divisor, int(filters + self.depth_divisor / 2) // self.depth_divisor * self.depth_divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += self.depth_divisor
        return int(new_filters)

    def round_repeats(self, repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(self.depth_coefficient * repeats))
    
    def block(self,
              inputs,
              activation='swish',
              drop_rate=0.,
              name='',
              filters_in=32,
              filters_out=16,
              kernel_size=3,
              strides=1,
              expand_ratio=1,
              se_ratio=0.,
              id_skip=True):
        """An inverted residual block.

            Arguments:
                inputs: input tensor.
                activation: activation function.
                drop_rate: float between 0 and 1, fraction of the input units to drop.
                name: string, block label.
                filters_in: integer, the number of input filters.
                filters_out: integer, the number of output filters.
                kernel_size: integer, the dimension of the convolution window.
                strides: integer, the stride of the convolution.
                expand_ratio: integer, scaling coefficient for the input filters.
                se_ratio: float between 0 and 1, fraction to squeeze the input filters.
                id_skip: boolean.

            Returns:
                output tensor for the block.
        """
        # Expansion phase
        filters = filters_in * expand_ratio
        if expand_ratio != 1:
            x = tf.keras.layers.Conv2D(
                filters,
                1,
                padding='same',
                use_bias=False,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + 'expand_conv')(inputs)
            x = build_normalization(**self.normalization, name=name + 'expand_bn')(x)
            x = build_activation(**self.activation, name=name + 'expand_activation')(x)
        else:
            x = inputs

        # Depthwise Convolution
        if strides == 2:
            x = tf.keras.layers.ZeroPadding2D(
                padding=imagenet_utils.correct_pad(x, kernel_size),
                name=name + 'dwconv_pad')(x)
            conv_pad = 'valid'
        else:
            conv_pad = 'same'
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size,
            strides=strides,
            padding=conv_pad,
            use_bias=False,
            depthwise_initializer=CONV_KERNEL_INITIALIZER,
            name=name + 'dwconv')(x)
        x = build_normalization(**self.normalization, name=name + 'bn')(x)
        x = build_activation(**self.activation, name=name + 'activation')(x)

        # Squeeze and Excitation phase
        if 0 < se_ratio <= 1:
            filters_se = max(1, int(filters_in * se_ratio))
            se = tf.keras.layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
            se = tf.keras.layers.Reshape((1, 1, filters), name=name + 'se_reshape')(se)
            se = tf.keras.layers.Conv2D(
                filters_se,
                1,
                padding='same',
                activation=activation["activation"],
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + 'se_reduce')(se)
            se = tf.keras.layers.Conv2D(
                filters,
                1,
                padding='same',
                activation='sigmoid',
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + 'se_expand')(se)
            x = tf.keras.layers.multiply([x, se], name=name + 'se_excite')

        # Output phase
        x = tf.keras.layers.Conv2D(
            filters_out,
            1,
            padding='same',
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + 'project_conv')(x)
        x = build_normalization(**self.normalization, name=name + 'project_bn')(x)
        if id_skip and strides == 1 and filters_in == filters_out:
            if drop_rate > 0:
                x = tf.keras.layers.Dropout(
                    drop_rate, noise_shape=(None, 1, 1, 1), name=name + 'drop')(x)
                x = tf.keras.layers.add([x, inputs], name=name + 'add')
        return x

    def build_model(self):
        # Build stem
        x = self.img_input
        x = layers.Rescaling(1. / 255.)(x)
        x = layers.Normalization(axis=self.bn_axis)(x)

        x = tf.keras.layers.ZeroPadding2D(
            padding=imagenet_utils.correct_pad(x, 3),
            name='stem_conv_pad')(x)
        x = tf.keras.layers.Conv2D(
            self.round_filters(32),
            3,
            strides=2,
            padding='valid',
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name='stem_conv')(x)
   
        x = build_normalization(**self.normalization, name='stem_bn')(x)
        x = build_activation(**self.activation, name='stem_activation')(x)

        block_outputs = [x]
        # Build blocks
        blocks_args = copy.deepcopy(self.blocks_args)
        b = 0
        blocks = float(sum(args['repeats'] for args in blocks_args))
        output_stages = [1, 2, 4, 6]
        for (i, args) in enumerate(blocks_args):
            assert args['repeats'] > 0
            # Update block input and output filters based on depth multiplier.
            args['filters_in'] = self.round_filters(args['filters_in'])
            args['filters_out'] = self.round_filters(args['filters_out'])

            strides = args["strides"]

            for j in range(self.round_repeats(args.pop('repeats'))):
                # The first block needs to take care of stride and filter size increase.
                if j > 0:
                    args['strides'] = 1
                    args['filters_in'] = args['filters_out']
                x = self.block(
                    x,
                    self.activation,
                    self.drop_connect_rate * b / blocks,
                    name='block{}{}_'.format(i + 1, chr(j + 97)),
                    **args)
                b += 1
            if i in output_stages:
                block_outputs.append(x)

        # Build top
        x = tf.keras.layers.Conv2D(
            self.round_filters(1280),
            1,
            padding='same',
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name='top_conv')(x)
        x = build_normalization(**self.normalization, name='top_bn')(x)
        x = build_activation(**self.activation, name='top_activation')(x)
        if -1 in self.output_indices:
            x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
            if self.dropout_rate > 0:
                x = tf.keras.layers.Dropout(self.dropout_rate, name='top_dropout')(x)
            x = tf.keras.layers.Dense(1000,
                                      activation=self.classifier_activation,
                                      kernel_initializer=DENSE_KERNEL_INITIALIZER,
                                      name='predictions')(x)
            # Ensure that the model takes into account
            # any potential predecessors of `input_tensor`.
            if self.input_tensor is not None:
                inputs = tf.keras.utils.get_source_inputs(self.input_tensor)
            else:
                inputs = self.img_input
            # Create model.
            return tf.keras.Model(inputs=self.img_input, outputs=x, name=model_name)
   
        return [block_outputs[i - 1] for i in self.output_indices]


if __name__ == "__main__":
    model_name = "efficientnetb3"
    shape = 300 # 380
    efficientnet = EfficientNet(model_name,
                                output_indices=[3, 4, 5],
                                frozen_stages=(0, 1, 2),

                                normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=False))

    checkpoint_dir = "/home/bail/Workspace/pretrained_weights/%s.h5" % model_name
    stages = efficientnet.build_model()
    model = tf.keras.Model(inputs=efficientnet.img_input, outputs=stages)
    model.load_weights(checkpoint_dir, by_name=True)

    with tf.io.gfile.GFile("/home/bail/Documents/p3.jpeg", "rb") as gf:
        images = tf.image.decode_jpeg(gf.read())

    images = tf.image.resize(images, (shape, shape))[None]

    cls = model(images, training=False)
    # tf.print(tf.nn.top_k(tf.squeeze(cls), k=5))

    for variable in model.trainable_variables:
        tf.print(variable.name)
        # if "bias" in variable.name:
        #     print(variable)
    