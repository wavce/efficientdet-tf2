import tensorflow as tf


class PSRoIPooling(tf.keras.layers.Layer):
    """Position-sensitive crop and pool rectangular regions from a feature grid.

        The output crops are split into `spatial_bins_y` vertical bins
        and `spatial_bins_x` horizontal bins. For each intersection of a vertical
        and a horizontal bin the output values are gathered by performing
        `tf.image.crop_and_resize` (bilinear resampling) on a a separate subset of
        channels of the image. This reduces `depth` by a factor of
        `(spatial_bins_y * spatial_bins_x)`.

        When global_pool is True, this function implements a differentiable version
        of position-sensitive RoI pooling used in
        [R-FCN detection system](https://arxiv.org/abs/1605.06409).

        When global_pool is False, this function implements a differentiable version
        of position-sensitive assembling operation used in

        [instance FCN](https://arxiv.org/abs/1603.08678).

        Args:
            crop_size: A list of two integers `[crop_height, crop_width]`. All
                cropped image patches are resized to this size. The aspect ratio of the
                image content is not preserved. Both `crop_height` and `crop_width` need
                to be positive.

            num_spatial_bins: A list of two integers `[spatial_bins_y, spatial_bins_x]`.
                Represents the number of position-sensitive bins in y and x directions.
                Both values should be >= 1. `crop_height` should be divisible by
                `spatial_bins_y`, and similarly for width.

                The number of image channels should be divisible by
                (spatial_bins_y * spatial_bins_x).

                Suggested value from R-FCN paper: [3, 3].

            global_pool: A boolean variable.
                If True, we perform average global pooling on the features assembled from
                    the position-sensitive score maps.

                If False, we keep the position-pooled features without global pooling
                    over the spatial coordinates.

                Note that using global_pool=True is equivalent to but more efficient than
                running the function with global_pool=False and then performing global
                average pooling.

        Raises:
            ValueError: Raised in four situations:
                `num_spatial_bins` is not >= 1;
                `num_spatial_bins` does not divide `crop_size`;
                 `(spatial_bins_y*spatial_bins_x)` does not divide `depth`;
                `bin_crop_size` is not square when global_pool=False due to the
                    constraint in function space_to_depth.
    """
    def __init__(self,
                 num_boxes,
                 crop_size=(9, 9),
                 num_spatial_bins=(3, 3),
                 global_pool=True,
                 **kwargs):
        super(PSRoIPooling, self).__init__(**kwargs)

        total_bins = 1
        bin_crop_size = []
        for num_bins, crop_dim in zip(num_spatial_bins, crop_size):
            if num_bins < 1:
                raise ValueError("num_spatial_bins should be >= 1.")

            if crop_dim % num_bins != 0:
                raise ValueError("crop_size should be divisible by num_spatial_bins.")

            total_bins *= num_bins
            bin_crop_size.append(crop_dim // num_bins)

        if not global_pool and bin_crop_size[0] != bin_crop_size[1]:
            raise ValueError("Only support square bin crop size for now.")

        self.num_spatial_bins = num_spatial_bins
        self.bin_crop_size = bin_crop_size
        self.total_bins = total_bins
        self.global_pool = global_pool

        self.num_boxes = num_boxes

    def build(self, input_shape):
        super(PSRoIPooling, self).build(input_shape)

    def one_image_pooling(self, image, boxes):
        """
        Args:
            image: A `Tensor`. Must be one of the following types: `uint8`, `int8`,
            `int16`, `int32`, `int64`, `half`, `float32`, `float64`.
            A 3-D tensor of shape `[image_height, image_width, depth]`.
            Both `image_height` and `image_width` need to be positive.

            boxes: A `Tensor` of type `float32`.
                A 2-D tensor of shape `[num_boxes, 4]`. Each box is specified in
                normalized coordinates `[y1, x1, y2, x2]`. A normalized coordinate value
                of `y` is mapped to the image coordinate at `y * (image_height - 1)`, so
                as the `[0, 1]` interval of normalized image height is mapped to
                `[0, image_height - 1] in image height coordinates. We do allow y1 > y2,
                in which case the sampled crop is an up-down flipped version of the
                original image. The width dimension is treated similarly.

        Returns:
            position_sensitive_features: A 4-D tensor of shape `[num_boxes, crop_channels]`,
            where `crop_channels = depth / (spatial_bins_y * spatial_bins_x)`.
        """
        y1, x1, y2, x2 = tf.unstack(boxes, axis=1)
        spatial_bins_y, spatial_bins_x = self.num_spatial_bins[0], self.num_spatial_bins[1]

        position_sensitive_boxes = tf.TensorArray(size=self.total_bins, dtype=boxes.dtype)
        i = tf.constant(0, dtype=tf.int32)
        for bin_y in tf.range(spatial_bins_y):
            step_y = (y2 - y1) / tf.cast(spatial_bins_y, boxes.dtype)
            for bin_x in tf.range(spatial_bins_x):
                step_x = (x2 - x1) / tf.cast(spatial_bins_x, boxes.dtype)
                i += tf.constant(1., dtype=tf.int32)
                box_coord = tf.convert_to_tensor([y1 + bin_y * step_y,
                                                  x1 + bin_x * step_x,
                                                  y1 + (bin_y + 1) * step_y,
                                                  x1 + (bin_x + 1) * step_x], dtype=boxes.dtype)
                position_sensitive_boxes.write(i, box_coord)

        split_step = tf.shape(image)[2] // self.total_bins

        image_crops = tf.TensorArray(size=self.total_bins, dtype=image.dtype)
        for i in tf.range(self.total_bins):
            split = image[..., (i*split_step):((i+1)*split_step)]
            box = position_sensitive_boxes.read(i)

            crop = tf.image.crop_and_resize(tf.expand_dims(split, 0),
                                            boxes=box,
                                            box_indices=tf.zeros(tf.shape(boxes)[0], dtype=tf.int32),
                                            crop_size=self.bin_crop_size)
            image_crops.write(i, crop)

        image_crops = image_crops.stack(axis=0)   # 5-D

        position_sensitive_features = tf.reduce_mean(image_crops, axis=0)
        position_sensitive_features = tf.reduce_mean(position_sensitive_features, [1, 2], keepdims=True)

        position_sensitive_boxes.close()
        image_crops.close()

        return position_sensitive_features

    def call(self, inputs, boxes):
        batch_size = tf.shape(inputs)[0]

        position_sensitive_features = tf.TensorArray(size=batch_size, dtype=inputs.dtype)
        for b in tf.range(batch_size):
            features = self.one_image_pooling(inputs[b], boxes[b])
            position_sensitive_features.write(b, features)

        return position_sensitive_features.stack(axis=0)

    def compute_output_shape(self, input_shape):
        cropped_channels = input_shape[-1] // self.total_bins
        return tf.TensorShape([input_shape[0], self.num_boxes, cropped_channels])

    def get_config(self):
        config = {"num_boxes": self.num_boxes,
                  "bin_crop_size": self.bin_crop_size,
                  "num_spatial_bins": self.num_spatial_bins}

        base_config = super(PSRoIPooling, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
