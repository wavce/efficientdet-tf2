import tensorflow as tf


class Scale(tf.keras.layers.Layer):
    def __init__(self, value, **kwargs):
        super(Scale, self).__init__(**kwargs)

        self.value = value
        self.scale = None

    @property
    def _param_dtype(self):
        if self.dtype == tf.float16 or self.dtype == tf.bfloat16:
            return tf.float32
        else:
            return self.dtype or tf.float32

    def build(self, input_shape):
        self.scale = self.add_weight(name="scale",
                                     shape=[],
                                     dtype=self._param_dtype,
                                     initializer=tf.keras.initializers.Ones(),
                                     experimental_autocast=False)

    def call(self, inputs, **kwargs):
        x = tf.cast(inputs, self._param_dtype)
        x = tf.exp(x) * self.scale
        x = tf.cast(x, inputs.dtype)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "value": self.value
        }

        base_config = super(Scale, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class GroupNormalization(tf.keras.layers.Layer):
    """
    Group normalization.

    Args:
        axis: Integer, the axis that should be normalized
            (typically the features axis). For instance,
            after a `Conv2D` layer with `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        group: Group for split tensor.
        epsilon: Small float added to variance to avoid dividing by zero.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
    """
    def __init__(self,
                 axis=-1,
                 group=32,
                 epsilon=1e-5,
                 beta_initializer="zeros",
                 gamma_initializer="ones",
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.group = group
        self.epsilon = epsilon

        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)

        self.gamma = None
        self.beta = None

        self.channels_last = self.axis == -1 or self.axis == 3

        self.spatial_dims = [1, 2, 3] if self.channels_last else [2, 3, 4]

    @property
    def _param_dtype(self):
        if self.dtype == tf.float16 or self.dtype == tf.bfloat16:
            return tf.float32
        else:
            return self.dtype or tf.float32

    def build(self, input_shape):
        if self.channels_last:
            c = input_shape[-1]
        else:
            c = input_shape[1]
        weight_shape = [c]

        self.gamma = self.add_weight(name="gamma",
                                     shape=weight_shape,
                                     dtype=self._param_dtype,
                                     initializer=self.gamma_initializer,
                                     experimental_autocast=False)
        self.beta = self.add_weight(name="beta",
                                    shape=weight_shape,
                                    dtype=self._param_dtype,
                                    initializer=self.beta_initializer,
                                    experimental_autocast=False)

    def call(self, inputs, **kwargs):
        shape = tf.shape(inputs)
        if self.channels_last:
            group_shape = [-1, shape[1], shape[2], shape[3] // self.group, self.group]
        else:
            group_shape = [-1, self.group, shape[1] // self.group, shape[2], shape[3]]
        inputs = tf.reshape(inputs, group_shape)

        x = tf.cast(inputs, self._param_dtype)
        mean, variance = tf.nn.moments(x, axes=self.spatial_dims, keepdims=True)
        variance = tf.cast(variance, inputs.dtype)
        mean = tf.cast(mean, inputs.dtype)
        
        inv = tf.math.rsqrt(variance + self.epsilon)  # 1. / sqrt(v)

        outputs = (inputs - mean) * inv
        outputs = tf.reshape(outputs, shape)
        outputs = outputs * self.gamma + self.beta

        return outputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

    def get_config(self):
        config = {"axis": self.axis,
                  "epsilon": self.epsilon,
                  "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
                  "gamma_initializer": tf.keras.initializers.serialize(self.gamma_initializer)}

        base_config = super(GroupNormalization, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class L2Normalization(tf.keras.layers.Layer):
    def __init__(self, value, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)

        self._value = value

    @property
    def _param_dtype(self):
        if self.dtype == tf.float16 or self.dtype == tf.bfloat16:
            return tf.float32
        else:
            return self.dtype or tf.float32

    def build(self, input_shape):
        self.scale = self.add_weight(name="scale",
                                     shape=[],
                                     dtype=self.dtype,
                                     initializer=tf.keras.initializers.Constant(self._value))
        super(L2Normalization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = tf.nn.l2_normalize(inputs)
        x = tf.math.scalar_mul(self.scale, x)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"value": self._value}

        base_config = super(L2Normalization, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class SwitchableNormalization(tf.keras.layers.Layer):
    def __init__(self,
                 axis=-1,
                 epsilon=1e-7,
                 beta_initializer=tf.keras.initializers.Ones(),
                 gamma_initializer=tf.keras.initializers.Zeros(),
                 **kwargs):
        super(SwitchableNormalization, self).__init__(**kwargs)

        # assert axis == -1 or axis == 3, "SwitchableNormalization only supports channels last (axis=-1 or axis=3)."
        channels_last = axis in [-1, 3]
        self._bn_spatial_dims = [0, 1, 2] if channels_last else [0, 2, 3]
        self._in_spatial_dims = [1, 2]
        self._ln_spatial_dims = [1, 2, 3]
        self.axis = axis
        self.epsilon = epsilon
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer

    @property
    def _param_dtype(self):
        if self.dtype == tf.float16 or self.dtype == tf.bfloat16:
            return tf.float32
        else:
            return self.dtype or tf.float32
    
    def build(self, input_shape):
        weight_shape = [input_shape[self.axis]]

        self.gamma = self.add_weight(name="gamma",
                                     shape=weight_shape,
                                     dtype=self._param_dtype,
                                     initializer=self.gamma_initializer,
                                     experimental_autocast=False)
        self.beta = self.add_weight(name="beta",
                                    shape=weight_shape,
                                    dtype=self._param_dtype,
                                    initializer=self.beta_initializer,
                                    experimental_autocast=False)

        self.mean_weights = self.add_weight(name="mean_weights",
                                            shape=[3],
                                            dtype=self.dtype,
                                            initializer=tf.keras.initializers.Ones(),
                                            experimental_autocast=False)
        self.var_weights = self.add_weight(name="var_weights",
                                           shape=[3],
                                           dtype=self.dtype,
                                           initializer=tf.keras.initializers.Ones(),
                                           experimental_autocast=False)

    def call(self, inputs):
        x = tf.cast(inputs, self._param_dtype)
        batch_mean, batch_var = tf.nn.moments(x, axes=self._bn_spatial_dims, keepdims=True)
        instance_mean, instance_var = tf.nn.moments(x, axes=self._in_spatial_dims, keepdims=True)
        layer_mean, layer_var = tf.nn.moments(x, axes=self._ln_spatial_dims, keepdims=True)

        mean_weights = tf.nn.softmax(self.mean_weights)
        var_weights = tf.nn.softmax(self.var_weights)
        mean_weights = tf.cast(mean_weights, inputs.dtype)
        var_weights = tf.cast(var_weights, inputs.dtype)
        batch_mean = tf.cast(batch_mean, inputs.dtype)
        batch_var = tf.cast(batch_var, inputs.dtype)
        instance_mean = tf.cast(instance_mean, inputs.dtype)
        instance_var = tf.cast(instance_var, inputs.dtype)
        layer_mean = tf.cast(layer_mean, inputs.dtype)
        layer_var = tf.cast(layer_var, inputs.dtype)
        gamma = tf.cast(self.gamma, inputs.dtype)
        beta = tf.cast(self.beta, inputs.dtype)

        mean = mean_weights[0] * batch_mean + mean_weights[1] * instance_mean + mean_weights[2] * layer_mean
        var = var_weights[0] * batch_var + var_weights[1] * instance_var + var_weights[2] * layer_var

        outputs = (inputs - mean) * tf.math.rsqrt(var + self.epsilon)
        outputs = outputs * gamma + beta

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"axis": self.axis,
                  "epsilon": self.epsilon,
                  "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
                  "gamma_initializer": tf.keras.initializers.serialize(self.gamma_initializer)}

        base_config = super(SwitchableNormalization, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class FilterResponseNormalization(tf.keras.layers.Layer):
    def __init__(self,
                 axis=-1,
                 epsilon=1e-6,
                 gamma_initializer=tf.keras.initializers.Ones(),
                 beta_initializer=tf.keras.initializers.Zeros(),
                 tau_initializer=tf.keras.initializers.Zeros(),
                 use_fixed_epsilon=True,
                 **kwargs):
        super(FilterResponseNormalization, self).__init__(**kwargs)

        assert axis == -1 or axis == 3, "FilterResponseNormalization only supports channels last (axis=-1 or axis=3)."

        self.axis = axis
        self.epsilon = epsilon
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.tau_initializer = tau_initializer

        self._use_fixed_eps = use_fixed_epsilon

    @property
    def _param_dtype(self):
        # Raise parameters of fp16 batch norm to fp32
        if self.dtype == tf.float16 or self.dtype == tf.bfloat16:
            return tf.float32

        return self.dtype or tf.float32

    def build(self, input_shape):
        weight_shape = [input_shape[self.axis]]

        self.gamma = self.add_weight(name="gamma",
                                     shape=weight_shape,
                                     dtype=self._param_dtype,
                                     initializer=self.gamma_initializer,
                                     experimental_autocast=False)
        self.beta = self.add_weight(name="beta",
                                    shape=weight_shape,
                                    dtype=self._param_dtype,
                                    initializer=self.beta_initializer,
                                    experimental_autocast=False)
        self.tau = self.add_weight(name="tau",
                                   shape=weight_shape,
                                   dtype=self._param_dtype,
                                   initializer=self.tau_initializer,
                                   experimental_autocast=False)
        if not self._use_fixed_eps:
            self.epsilon = self.add_weight(name="epsilon",
                                           shape=[],
                                           dtype=self._param_dtype,
                                           initializer=tf.keras.initializers.Constant(self.epsilon),
                                           experimental_autocast=False)

    def call(self, inputs, training=None):
        # Compute the mean norm of activations per channel.
        nu2 = tf.reduce_mean(tf.math.square(tf.cast(inputs, self._param_dtype)), axis=[1, 2], keepdims=True)
        nu2 = tf.cast(nu2, inputs.dtype)
        # Perform FRN
        x = tf.math.rsqrt(nu2 + tf.math.abs(tf.convert_to_tensor(self.epsilon, dtype=inputs.dtype)))
        x = inputs * x
        gamma = tf.cast(self.gamma, inputs.dtype)
        beta = tf.cast(self.beta, inputs.dtype)
        tau = tf.cast(self.tau, inputs.dtype)
        # Applying the Offset-ReLU non-linearity
        outputs = tf.math.maximum(gamma * x + beta, tau)

        return outputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

    def get_config(self):
        config = {"axis": self.axis,
                  "epsilon": self.epsilon if isinstance(self.epsilon, float) else None,
                  "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
                  "gamma_initializer": tf.keras.initializers.serialize(self.gamma_initializer),
                  "tau_initializer": tf.keras.initializers.serialize(self.tau_initializer)}

        base_config = super(FilterResponseNormalization, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
