import tensorflow as tf


class Swish(tf.keras.layers.Layer):
    """Computes the Swish activation function.
        We provide three alternnatives:
            - Native tf.nn.swish, use less memory during training than composable swish.
            - Quantization friendly hard swish.
            - A composable swish, equivalant to tf.nn.swish, but more general for
                finetuning and TF-Hub.
        Args:
            features: A `Tensor` representing preactivation values.
            use_native: Whether to use the native swish from tf.nn that uses a custom
                gradient to reduce memory usage, or to use customized swish that uses
                default TensorFlow gradient computation.
            use_hard: Whether to use quantization-friendly hard swish.
        Returns:
            The activation value.
    """
    def __init__(self, use_native=True, use_hard=False, **kwargs):
        super(Swish, self).__init__(**kwargs)

        if use_hard and use_native:
            raise ValueError("Cannot specify both use_native and use_hard.")

        if use_native:
            self._swish = self._native_swish
        elif use_hard:
            self._swish = self._hard_swish
        else:
            self._swish = self._cumstom_swish

    def _native_swish(self, inputs):
        return tf.nn.swish(inputs)

    def _hard_swish(self, inputs):
        return inputs * tf.nn.relu6(inputs + 3.) * (1./6.)

    def _cumstom_swish(self, inputs):
        return inputs * tf.nn.sigmoid(inputs)

    def call(self, inputs):
        return self._swish(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
