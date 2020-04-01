import tensorflow as tf
from tensorflow.python.ops import nn_ops
from tensorflow.python.keras.layers.convolutional import Conv


def _get_grid_offsets(input_shape, kernel_size, dtype=tf.float32):
    with tf.name_scope("get_grid_offsets"):
        input_h = input_shape[0]
        input_w = input_shape[1]
        n_kernel = kernel_size[0] * kernel_size[1]

        # initial_offsets => (kh, kw, 2)  kh means the height of kernel
        initial_offsets = tf.stack(tf.meshgrid(tf.range(kernel_size[0]),
                                               tf.range(kernel_size[1])))
        initial_offsets = tf.expand_dims(
            tf.expand_dims(tf.reshape(initial_offsets, (-1, 2)), axis=0), axis=0)  # (1, 1, kw * kh, 2)
        initial_offsets = tf.tile(initial_offsets, [input_h, input_w, 1, 1])  # (input_h, input_w, kw * kh, 2)
        initial_offsets = tf.cast(initial_offsets, dtype)

        grid_x, grid_y = tf.meshgrid(tf.range(-int((kernel_size[1] - 1) / 2),
                                              int(input_h - int((kernel_size[0] - 1) / 2)), 1),
                                     tf.range(-int((kernel_size[0] - 1) / 2),
                                              int(input_w - int((kernel_size[1] - 1) / 2)), 1))
        grid = tf.cast(tf.stack([grid_y, grid_x], axis=-1), dtype)  # (input_h, input_w, 2)
        grid = tf.tile(tf.expand_dims(grid, axis=2), [1, 1, n_kernel, 1])  # (input_h, input_w, kh * kw, 2)

        grid_offset = grid + initial_offsets

        return grid_offset


def _batch_map_coordinates(inputs, coords):
    """Batch map coordinates, only supports 2D feature maps.

        Here is the step-by-step algorithm:
        1. The multi-level features are gathered into a
            [batch_size, height, width, num_kernel * 2, num_filters]
            Tensor. The Tensor contains four neighboring feature points for each
            vertices in the output grid.
        2. Compute the interpolation kernel of shape
            [batch_size, height, width, num_kernel * 2]. The last height and width axis
            can be seen as stacking 2x2 interpolation kernels for all vertices in the
            output grid.
        3. Element-wise multiply the gathered features and interpolation kernel.
            Then apply 1x2 average pooling to reduce last dimension to
            output_size.

    Args:
        inputs: A Tensor, has shape [batch_size, h, w, filters]
        coords: A Tensor, has shape [batch_size, h, w, kh * hw, 2], kh is the height of kernel.

    Returns:
        A Tensor with shape [batch_size, h, w, kh * kw, filters]
    """
    with tf.name_scope("batch_map_coordinates"):
        batch_size = tf.shape(inputs)[0]
        input_h, input_w = tf.shape(inputs)[1], tf.shape(inputs)[2]
        n_kernel_size = tf.shape(coords)[3]
        num_filters = tf.shape(inputs)[3]
        n_coords = input_h * input_w * n_kernel_size
        coords_y0 = tf.math.floor(coords[..., 0])
        coords_x0 = tf.math.floor(coords[..., 1])
        coords_y1 = tf.math.ceil(coords[..., 0])
        coords_x1 = tf.math.ceil(coords[..., 1])

        y_indices = tf.cast(tf.reshape(tf.stack([coords_y0, coords_y1], axis=-1),
                                       [batch_size, input_h, input_w, n_kernel_size * 2]),
                            tf.int32)  # (b, h, w, n_kernel * 2)
        x_indices = tf.cast(tf.reshape(tf.stack([coords_x0, coords_x1], axis=-1),
                                       [batch_size, input_h, input_w, n_kernel_size * 2]),
                            tf.int32)  # (b, h, w, n_kernel * 2)

        indices = tf.reshape(tf.tile(tf.reshape(tf.range(batch_size) * n_coords,
                                                [batch_size, 1, 1, 1]),
                                     [1, input_h, input_w, n_kernel_size * 2]) +
                             y_indices * input_w * n_kernel_size + x_indices * n_kernel_size +
                             tf.tile(tf.reshape(tf.range(n_kernel_size),
                                                [1, 1, 1, n_kernel_size]),
                                     [batch_size, input_h, input_w, 2]), [-1])

        inputs = tf.tile(tf.expand_dims(inputs, axis=3), [1, 1, 1, n_kernel_size, 1])
        features = tf.gather(tf.reshape(inputs, [-1, num_filters]), indices)
        features = tf.reshape(features, [batch_size, input_h, input_w, n_kernel_size * 2, num_filters])
        # The RoIAlign feature f can be computed by bilinear interpolation of four
        # neighboring feature points f0, f1, f2, and f3.
        # f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
        #                       [f10, f11]]
        # f(y, x) = (hy*hx)f00 + (hy*lx)f01 + (ly*hx)f10 + (lx*ly)f11
        # f(y, x) = w00*f00 + w01*f01 + w10*f10 + w11*f11
        ly = coords[..., 0] - coords_y0  # (b, h, w, n_kernel)
        lx = coords[..., 1] - coords_x0
        kernel_y = tf.reshape(tf.stack([1. - ly, ly], axis=4), [batch_size, input_h, input_w, n_kernel_size * 2])
        kernel_x = tf.reshape(tf.stack([1. - lx, lx], axis=4), [batch_size, input_h, input_w, n_kernel_size * 2])
        interpolation_kernel = kernel_y * kernel_x * 2
        # Interpolates the gathered features with computed interpolation kernels.
        features *= tf.cast(tf.expand_dims(interpolation_kernel, axis=4), dtype=features.dtype)
        features = tf.nn.avg_pool(features, [1, 1, 1, 2, 1], [1, 1, 1, 2, 1], "VALID")

        return features


class DeformableConv2D(Conv):
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
        if isinstance(kernel_size, int):
            self.n_kernel = int(kernel_size * kernel_size)
        else:
            self.n_kernel = int(kernel_size[0] * kernel_size[1])
        self.initial_kernel_size = (kernel_size
                                    if isinstance(kernel_size, (list, tuple))
                                    else [kernel_size, kernel_size])
        kernel_size = [1, 1, self.n_kernel]
        strides = (list(strides) + [1]
                   if isinstance(strides, (list, tuple))
                   else [strides, strides, strides])
        dilation_rate = (list(dilation_rate) + [1]
                         if isinstance(dilation_rate, (list, tuple))
                         else [dilation_rate, dilation_rate, dilation_rate])
        # if data_format is not None:
        #     assert data_format == "channels_last"

        super(DeformableConv2D, self).__init__(rank=3,
                                               filters=filters,
                                               kernel_size=kernel_size,
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
        self.input_spec = [tf.keras.layers.InputSpec(ndim=4), tf.keras.layers.InputSpec(ndim=4)]

    def build(self, input_shape):
        input_shape = input_shape[0]
        self.grid_offsets = _get_grid_offsets(input_shape[1:3], self.initial_kernel_size, dtype=self.dtype)

        input_shape = tf.TensorShape([input_shape[0],
                                      input_shape[1],
                                      input_shape[1],
                                      self.n_kernel * 2,
                                      input_shape[-1]])
        super(DeformableConv2D, self).build(input_shape)

    def call(self, inputs):
        # Check if the input_shape in call() is different from that in build().
        # If they are different, recreate the _get_grid_offsets to avoid the stateful behavior.
        inputs, offset_layer = inputs
        call_input_shape = inputs.get_shape()
        recreate_conv_op = (
                call_input_shape[1:] != self._build_conv_op_input_shape[1:])

        if recreate_conv_op:
            self.grid_offsets = _get_grid_offsets(tf.shape(inputs)[1:3],
                                                  self.initial_kernel_size,
                                                  dtype=self.dtype)
        inputs = self._batch_map_offsets(inputs, self.grid_offsets, offset_layer)

        outputs = super(DeformableConv2D, self).call(inputs)
        outputs = tf.squeeze(outputs, 3)

        return outputs

    def _batch_map_offsets(self, features, grid_offsets, predicted_offsets):
        """Batch map offsets into inputs.

        Args:
            features: A Tensor, has shape (b, h, w, c).
            grid_offsets: A Tensor, the initial grid offsets, has shape (h, w, n_filters, 2)
            predicted_offsets: A Tensor, the predicted offsets, has shape (b, h, w, n_filters * 2)

        Returns:
            A Tensor has shape (b, h, w, n_filters * 2, c)
        """
        with tf.name_scope("batch_map_offsets"):
            b, h, w = tf.shape(features)[0], tf.shape(features)[1], tf.shape(features)[2]
            coords = tf.expand_dims(grid_offsets, 0)  # (1, h, w, n, 2)
            coords += tf.reshape(predicted_offsets, [b, h, w, self.n_kernel, 2])  # (b, h, w, n, 2)

            coords = tf.stack([tf.clip_by_value(coords[:, :, :, :, 0],
                                                0.0, tf.cast(h - 1, self.dtype)),
                               tf.clip_by_value(coords[:, :, :, :, 1],
                                                0.0, tf.cast(w - 1, self.dtype))], axis=-1)

            features = _batch_map_coordinates(features, coords)

            return features

    def compute_output_shape(self, input_shape):
        super(DeformableConv2D, self).comput_output_shape(input_shape[0])

    def get_config(self):
        config = {"initial_kernel_size": self.initial_kernel_size,}
        base_config = super(DeformableConv2D, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


def main():
    kernel_size = (3, 3)
    inputs = tf.random.uniform([2, 28, 28, 3])

    offsets = tf.random.uniform([2, 28, 28, 2 * kernel_size[0] * kernel_size[1]]) * 0

    conv = DeformableConv2D(filters=2,
                            kernel_size=kernel_size)
    outputs = conv([inputs, offsets])
    print(outputs.shape)


if __name__ == '__main__':
    main()

