import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.engine.input_spec import InputSpec


class WSConv2D(tf.keras.layers.Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(WSConv2D, self).__init__(filters,
                                       kernel_size,
                                       strides=strides,
                                       padding=padding,
                                       data_format=data_format,
                                       dilation_rate=dilation_rate,
                                       activation=activation,
                                       use_bias=use_bias,
                                       kernel_initializer=kernel_initializer,
                                       bias_initializer=bias_initializer,
                                       kernel_regularizer=kernel_regularizer,
                                       bias_regularizer=bias_regularizer,
                                       activity_regularizer=activity_regularizer,
                                       kernel_constraint=kernel_constraint,
                                       bias_constraint=bias_constraint,
                                       **kwargs)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None

        mean, variance = tf.nn.moments(self.kernel.value(), [0, 1, 2], keepdims=True)
        self.kernel.assign_sub(mean)
        self.kernel.assign(self.kernel.value() / (tf.sqrt(variance) + 1e-5))

        self.input_spec = tf.keras.backend.InputSpec(ndim=self.rank + 2,
                                                     axes={channel_axis: input_dim})
        self._convolution_op = tf.nn.Convolution(input_shape,
                                                 filter_shape=self.kernel.shape,
                                                 dilation_rate=self.dilation_rate,
                                                 strides=self.strides,
                                                 padding=self._get_padding_op(),
                                                 data_format=conv_utils.convert_data_format(self.data_format,
                                                                                            self.rank + 2))
        self.built = True

    def call(self, inputs):
        return super(WSConv2D, self).call(inputs)

    def compute_output_shape(self, input_shape):
        return super(WSConv2D, self).compute_output_shape(input_shape)
