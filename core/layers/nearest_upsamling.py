import tensorflow as tf


class NearestUpsampling2D(tf.keras.layers.Layer):
    """Nearest neighbor upsampling implementation.

    Args:
        scale: An integer multiple to scale resolution of input data.
    """
    def __init__(self, scale, **kwargs):
        super(NearestUpsampling2D, self).__init__(**kwargs)
        if "data_format" in kwargs:
            data_format = kwargs.pop("data_format")
            assert data_format in {"channels_first", "channels_last"}
            self.data_format = data_format 

        self.scale = scale

    def build(self, input_shape):
        super(NearestUpsampling2D, self).__init__(input_shape)

    def call(self, inputs, **kwargs):
        # Instead of broadcasting with a 6-d tensor, we're using stacking here
        # for TfLite compatibity.
        bs, h, w, c = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]
        # bs, h, w, c = inputs.get_shape().as_list()
        # bs = -1 if bs is None else bs
        # outputs = tf.stack([inputs] * self.scale, axis=3)
        # outputs = tf.stack([outputs] * self.scale, axis=2)
        scale = self.scale
        data = tf.reshape(inputs, [bs, h, 1, w, 1, c]) * tf.ones([1, 1, scale, 1, scale, 1], dtype=inputs.dtype)
        return tf.reshape(data, [bs, h * scale, w * scale, c])

    def compute_output_shape(self, input_shape):
        batch_size, h, w, c = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        return tf.TensorShape([batch_size, h * self.scale, w * self.scale, c])

