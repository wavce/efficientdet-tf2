import tensorflow as tf


class MaxInOut(tf.keras.layers.Layer):
    def __init__(self, num_negative, num_positive, axis=-1, **kwargs):
        super(MaxInOut, self).__init__(**kwargs)
        self.num_pos = num_positive
        self.num_neg = num_negative
        self.axis = axis

        self._max_in = num_negative > 1

    def build(self, input_shape):
        super(MaxInOut, self).__init__(input_shape)

    def call(self, inputs):
        neg, pos = tf.split(inputs, [self.num_neg, self.num_pos], self.axis)
        if self._max_in:
            neg = tf.reduce_max(neg, axis=self.axis, keepdims=True)
        else:
            pos = tf.reduce_max(pos, axis=self.axis, keepdims=True)

        outputs = tf.concat([neg, pos], axis=self.axis)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.axis == -1 or self.axis == 3:
            return tf.TensorShape([input_shape[0], input_shape[1], input_shape[2], 2])
        else:
            return tf.TensorShape([input_shape[0], 2, input_shape[2], input_shape[3]])

    def get_config(self):
        config = {
            'num_positive': self.num_pos,
            "num_negative": self.num_neg,
            "axis": self.axis
        }

        base_config = super(MaxInOut, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
