import tensorflow as tf


class DropBlock2D(tf.keras.layers.Layer):
    def __init__(self, block_size=7, keep_prob=0.9, data_format="channels_last", **kwargs):
        super(DropBlock2D, self).__init__(**kwargs)

        self.block_size = block_size
        self.keep_prob = keep_prob
        
        assert data_format in {"channels_first", "channels_last"}
        self.data_format = data_format

    def build(self, input_shape):
        super(DropBlock2D, self).build(input_shape)

    def _drop_block_nhwc(self, inputs):
        with tf.name_scope("drop_block_nhwc"):
            input_shape = tf.shape(inputs)
            top, left = input_shape[1] // 2, input_shape[2] // 2
            bottom, right = input_shape[1] - top, input_shape[2] - left
            padding = [[0, 0], [top, bottom], [left, right], [0, 0]]

            feat_size = tf.cast(input_shape[1:3], self.dtype)
            gamma1 = (1. - self.keep_prob) / (self.block_size * self.block_size)
            gamma2 = (feat_size[0] * feat_size[1]) / (feat_size[0] - self.block_size + 1.) / \
                     (feat_size[1] - self.block_size + 1.)

            gamma = gamma1 * gamma2

            mask_shape = [input_shape[0],
                          input_shape[1] - self.block_size + 1,
                          input_shape[2] - self.block_size + 1,
                          input_shape[3]]
            mask = tf.nn.relu(tf.sign(gamma - tf.random.uniform(mask_shape, 0, 1, dtype=self.dtype)))

            mask = tf.pad(mask, paddings=padding)
            mask = tf.nn.max_pool(mask, [1, self.block_size, self.block_size, 1], [1, 1, 1, 1], "SAME", "NHWC")

            mask = 1. - mask
            mask = mask * tf.cast(tf.size(mask), mask.dtype) / tf.reduce_sum(mask)
            mask = tf.cast(mask, inputs.dtype)
            outputs = mask * inputs

            return outputs
    
    def _drop_block_nchw(self, inputs):
        with tf.name_scope("drop_block_nchw"):
            input_shape = tf.shape(inputs)
            top, left = input_shape[2] // 2, input_shape[3] // 2
            bottom, right = input_shape[2] - top, input_shape[3] - left
            padding = [[0, 0], [0, 0], [top, bottom], [left, right]]

            feat_size = tf.cast(input_shape[2:], self.dtype)
            gamma1 = (1. - self.keep_prob) / (self.block_size * self.block_size)
            gamma2 = (feat_size[0] * feat_size[1]) / (feat_size[0] - self.block_size + 1.) / \
                     (feat_size[1] - self.block_size + 1.)

            gamma = gamma1 * gamma2

            mask_shape = [input_shape[0],
                          input_shape[1],
                          input_shape[2] - self.block_size + 1,
                          input_shape[3] - self.block_size + 1]
            mask = tf.nn.relu(tf.sign(gamma - tf.random.uniform(mask_shape, 0, 1, dtype=self.dtype)))

            mask = tf.pad(mask, paddings=padding)
            mask = tf.nn.max_pool(mask, [1, self.block_size, self.block_size, 1], [1, 1, 1, 1], "SAME", "NCHW")

            mask = 1. - mask
            mask = mask * tf.cast(tf.size(mask), mask.dtype) / tf.reduce_sum(mask)
            mask = tf.cast(mask, inputs.dtype)
            outputs = mask * inputs

            return outputs

    def call(self, inputs, training=None):
        if training:
            if self.data_format == "channels_first":
                return self._drop_block_nchw(inputs)
                
            return self._drop_block_nhwc(inputs)

        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"block_size": self.block_size,
                  "keep_probability": self.keep_prob}

        base_config = super(DropBlock2D, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

