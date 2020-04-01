import tensorflow as tf


def selective_crop_and_resize(features, boxes, box_levels, boundaries, output_size=7, sample_offset=0.5, align=True):
    """Crop and resize boxes on a set of feature maps.

     Given multiple features maps indexed by different levels, and a set of boxes
     where each box is mapped to a certain level, it selectively crops and resizes
     boxes from the corresponding feature maps to generate the box features.

     We follow the ROIAlign technique (see https://arxiv.org/pdf/1703.06870.pdf,
     figure 3 for reference). Specifically, for each feature map, we select an
     (output_size, output_size) set of pixels corresponding to the box location,
     and then use bilinear interpolation to select the feature value for each
     pixel.

     For performance, we perform the gather and interpolation on all layers as a
     single operation. This is op the multi-level features are first stacked and
     gathered into [2*output_size, 2*output_size] feature points. Then bilinear
     interpolation is performed on the gathered feature points to generate
     [output_size, output_size] RoIAlign feature map.

     Here is the step-by-step algorithm:
        1. The multi-level features are gathered into a
            [batch_size, num_boxes, output_size*2, output_size*2, num_filters]
            Tensor. The Tensor contains four neighboring feature points for each
            vertices in the output grid.
        2. Compute the interpolation kernel of shape
            [batch_size, num_boxes, output_size*2, output_size*2]. The last 2 axis
            can be seen as stacking 2x2 interpolation kernels for all vertices in the
            output grid.
        3. Element-wise multiply the gathered features and interpolation kernel.
            Then apply 2x2 average pooling to reduce spatial dimension to
            output_size.
     Args:
        features: a 5-D tensor of shape
            [batch_size, num_levels, max_height, max_width, num_filters] where
            cropping and resizing are based.
        boxes: a 3-D tensor of shape [batch_size, num_boxes, 4] encoding the
            information of each box w.r.t. the corresponding feature map.
            boxes[:, :, 0:2] are the grid position in (y, x) (float) of the top-left
            corner of each box. boxes[:, :, 2:4] are the box sizes in (h, w) (float)
            in terms of the number of pixels of the corresponding feature map size.
        box_levels: a 3-D tensor of shape [batch_size, num_boxes, 1] representing
            the 0-based corresponding feature level index of each box.
        boundaries: a 3-D tensor of shape [batch_size, num_boxes, 2] representing
            the boundary (in (y, x)) of the corresponding feature map for each box.
            Any resampled grid points that go beyond the bounary will be clipped.
        output_size: a scalar indicating the output crop size.
        sample_offset: a float number in [0, 1] indicates the subpixel sample offset
            from grid point.
     Returns:
        features_per_box: a 5-D tensor of shape
            [batch_size, num_boxes, output_size, output_size, num_filters]
            representing the cropped features.
     """
    # Compute the grid position w.r.t the corresponding feature map.
    with tf.name_scope("selective_crop_and_resize"):
        feat_shape = tf.shape(features)
        batch_size = feat_shape[0]
        num_levels = feat_shape[1]
        max_feature_height = feat_shape[2]
        max_feature_width = feat_shape[3]
        num_filters = feat_shape[4]
        num_boxes = tf.shape(boxes)[1]

        if align:
            box_y = boxes[..., 0] - 0.5
            box_x = boxes[..., 1] - 0.5
        else:
            box_y = boxes[..., 0]
            box_x = boxes[..., 1]
        box_grid_x = tf.TensorArray(size=output_size, dtype=boxes.dtype)
        box_grid_y = tf.TensorArray(size=output_size, dtype=boxes.dtype)
        for i in tf.range(output_size):
            gy = box_y + (tf.cast(i, boxes.dtype) + sample_offset) * boxes[:, :, 2] / output_size
            gx = box_x + (tf.cast(i, boxes.dtype) + sample_offset) * boxes[:, :, 3] / output_size
            box_grid_y = box_grid_y.write(i, gy)
            box_grid_x = box_grid_x.write(i, gx)
        box_grid_y = box_grid_y.stack()
        box_grid_x = box_grid_x.stack()
        box_grid_y = tf.transpose(box_grid_y, [1, 2, 0])
        box_grid_x = tf.transpose(box_grid_x, [1, 2, 0])

        box_grid_y0 = tf.maximum(0., tf.math.floor(box_grid_y))
        box_grid_x0 = tf.maximum(0., tf.math.floor(box_grid_x))
        box_grid_y0y1 = tf.stack([tf.minimum(box_grid_y0, tf.expand_dims(boundaries[:, :, 1], -1)),
                                  tf.minimum(box_grid_y0 + 1, tf.expand_dims(boundaries[:, :, 1], -1))], axis=3)
        box_grid_x0x1 = tf.stack([tf.minimum(box_grid_x0, tf.expand_dims(boundaries[:, :, 0], -1)),
                                  tf.minimum(box_grid_x0 + 1, tf.expand_dims(boundaries[:, :, 0], -1))], axis=3)

        y_indices = tf.cast(tf.reshape(box_grid_y0y1, [batch_size, num_boxes, output_size * 2]), dtype=tf.int32)
        x_indices = tf.cast(tf.reshape(box_grid_x0x1, [batch_size, num_boxes, output_size * 2]), dtype=tf.int32)

        height_dim_offset = max_feature_width
        level_dim_offset = max_feature_height * height_dim_offset
        batch_dim_offset = num_levels * level_dim_offset

        indices = tf.reshape(
            tf.tile(tf.reshape(tf.range(batch_size) * batch_dim_offset, [batch_size, 1, 1, 1]),
                    [1, num_boxes, output_size * 2, output_size * 2]) +
            tf.tile(tf.reshape(box_levels * level_dim_offset, [batch_size, num_boxes, 1, 1]),
                    [1, 1, output_size * 2, output_size * 2]) +
            tf.tile(tf.reshape(y_indices * height_dim_offset, [batch_size, num_boxes, output_size * 2, 1]),
                    [1, 1, 1, output_size * 2]) +
            tf.tile(tf.reshape(x_indices, [batch_size, num_boxes, 1, output_size * 2]),
                    [1, 1, output_size * 2, 1]), [-1])
        features = tf.reshape(features, [-1, num_filters])
        features_per_box = tf.reshape(tf.gather(features, indices),
                                      [batch_size, num_boxes, output_size * 2, output_size * 2, num_filters])
        # The RoIAlign feature f can be computed by bilinear interpolation of four
        # neighboring feature points f0, f1, f2, and f3.
        # f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
        #                       [f10, f11]]
        # f(y, x) = (hy*hx)f00 + (hy*lx)f01 + (ly*hx)f10 + (lx*ly)f11
        # f(y, x) = w00*f00 + w01*f01 + w10*f10 + w11*f11
        ly = box_grid_y - box_grid_y0
        lx = box_grid_x - box_grid_x0
        hy = 1.0 - ly
        hx = 1.0 - lx
        kernel_x = tf.reshape(tf.stack([hx, lx], axis=3),
                              [batch_size, num_boxes, 1, output_size * 2])
        kernel_y = tf.reshape(tf.stack([hy, ly], axis=3),
                              [batch_size, num_boxes, output_size * 2, 1])
        # Uses implicit broadcast to generate the interpolation kernel.
        # The multiplier `4` is for avg pooling.
        interpolation_kernel = kernel_y * kernel_x * 4
        # Interpolates the gathered features with computed interpolation kernels.
        features_per_box *= tf.cast(tf.expand_dims(interpolation_kernel, axis=4),
                                    dtype=features_per_box.dtype)
        features_per_box = tf.reshape(features_per_box,
                                      [batch_size * num_boxes, output_size * 2, output_size * 2, num_filters])
        features_per_box = tf.nn.avg_pool(features_per_box, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")
        features_per_box = tf.reshape(features_per_box,
                                      [batch_size, num_boxes, output_size, output_size, num_filters])

        return features_per_box


class MultiLevelAlignedRoIPooling(tf.keras.layers.Layer):
    """
    crop_size: A list of two integers `[crop_height, crop_width]`. All
      cropped image patches are resized to this size. The aspect ratio of the
      image content is not preserved. Both `crop_height` and `crop_width` need
      to be positive.
    """
    def __init__(self,
                 crop_size=7,
                 max_feature_shape=None,
                 anchor_strides=(8, 16, 32, 64, 128),
                 **kwargs):
        super(MultiLevelAlignedRoIPooling, self).__init__(**kwargs)

        self.crop_size = crop_size

        assert max_feature_shape is not None
        self.max_feature_height = max_feature_shape[0]
        self.max_feature_width = max_feature_shape[1]
        self.anchor_strides = anchor_strides

        self.num_levels = len(anchor_strides)
        self.min_level = min(anchor_strides) // 2
        self.max_level = max(anchor_strides) // 2

    # def _get_box_indices(self, proposals):
    #     proposal_shape = tf.shape(proposals)
    #
    #     indices = tf.ones(proposal_shape[0:2], dtype=tf.int32)
    #     multiplier = tf.expand_dims(tf.range(proposal_shape[0]), 1)
    #
    #     return tf.reshape(indices * multiplier, [-1])
    #
    # def single_level_aligned_roi_pooling(self, features, boxes):
    #     """
    #     Args:
    #         features: A `Tensor`. Must be one of the following types: `uint8`, `int8`,
    #         `int16`, `int32`, `int64`, `half`, 'bfloat16', `float32`, `float64`.
    #         A 4-D tensor of shape `[batch, height, width, depth]`.
    #
    #         boxes: A `Tensor` of type `float32`, 'bfloat16' or `float16`.
    #              A 3-D tensor of shape `[batch, num_boxes, 4]`. The boxes are specified in
    #             normalized coordinates and are of the form `[y1, x1, y2, x2]`. A
    #             normalized coordinate value of `y` is mapped to the image coordinate at
    #             `y * (image_height - 1)`, so as the `[0, 1]` interval of normalized image
    #             height is mapped to `[0, image_height - 1] in image height coordinates.
    #             The width dimension is treated similarly.
    #
    #     Returns:
    #         A 4-D tensor of shape `[num_boxes, crop_height, crop_width, depth]`
    #     """
    #     pooled_features = tf.image.crop_and_resize(image=features,
    #                                                boxes=tf.reshape(boxes, [-1, 4]),
    #                                                box_indices=self._get_box_indices(boxes),
    #                                                crop_size=self.crop_size)
    #     final_shape = tf.concat([tf.shape(boxes)[:2],
    #                              tf.shape(pooled_features)[1:]], axis=0)
    #     return tf.reshape(pooled_features, final_shape)

    def multi_levels_aligned_roi_pooling(self, features, boxes):
        """Crop and resize on multilevel feature pyramid.
           Generate the (output_size, output_size) set of pixels for each input box
           by first locating the box into the correct feature level, and then cropping
           and resizing it using the corresponding feature map of that level.

            Args:
                features: A list, The features are in shape of [batch_size, height, width, num_filters].
                boxes: A 3-D Tensor of shape [batch_size, num_boxes, 4]. Each row
                    represents a box with [y1, x1, y2, x2] in un-normalized coordinates.
            Returns:
                A 5-D tensor representing feature crop of shape
                [batch_size, num_boxes, output_size, output_size, num_filters].
        """
        with tf.name_scope("multi_levels_aligned_roi_pooling"):
            features_all = tf.stack([tf.image.pad_to_bounding_box(feat,
                                                                  offset_height=0,
                                                                  offset_width=0,
                                                                  target_height=self.max_feature_height,
                                                                  target_width=self.max_feature_width)
                                     for feat in features], axis=1)

            # Assigns boxes to right level
            box_height = boxes[:, :, 2] - boxes[:, :, 0]
            box_width = boxes[:, :, 3] - boxes[:, :, 1]
            area_sqrt = tf.math.sqrt(box_height * box_width)
            levels = tf.cast(tf.math.floordiv(tf.math.log(tf.math.divide(area_sqrt, 224.0)),
                                              tf.math.log(2.0)) + 4.0, dtype=tf.int32)
            # Maps levels between [min_level, max_level]
            levels = tf.minimum(self.max_level, tf.maximum(levels, self.min_level))
            # Projects box location and sizes to corresponding feature levels.
            scale_to_level = tf.cast(tf.math.pow(tf.constant(2.0), tf.cast(levels, tf.float32)),
                                     dtype=boxes.dtype)
            boxes /= tf.expand_dims(scale_to_level, axis=2)
            box_width /= scale_to_level
            box_height /= scale_to_level
            boxes = tf.concat([boxes[:, :, 0:2],
                               tf.expand_dims(box_height, -1),
                               tf.expand_dims(box_width, -1)], axis=-1)
            # Maps levels to [0, max_level-min_level]
            levels -= self.min_level
            level_strides = tf.pow([[2.0]], tf.cast(levels, tf.float32))
            boundary = tf.cast(tf.concat([tf.expand_dims([[tf.cast(self.max_feature_height, tf.float32)]]
                                                         / level_strides - 1, axis=-1),
                                          tf.expand_dims([[tf.cast(self.max_feature_width, tf.float32)]]
                                                         / level_strides - 1, axis=-1)], axis=-1), dtype=boxes.dtype)
            return selective_crop_and_resize(features_all, boxes, levels, boundary, self.crop_size)

    def call(self, inputs, proposals):
        return self.multi_levels_aligned_roi_pooling(inputs, proposals)

    def build(self, input_shape):
        super(MultiLevelAlignedRoIPooling, self).__init__(input_shape)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0],
                               self.num_proposals,
                               self.crop_size,
                               self.crop_size,
                               input_shape[-1]])

    def get_config(self):
        config = {"num_proposals": self.num_proposals, "crop_size": self.crop_size}

        base_config = super(MultiLevelAlignedRoIPooling, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


def main():
    pooling = MultiLevelAlignedRoIPooling(7, [256, 256], [4, 8, 16, 32, 64])
    features = [tf.random.uniform([1, 1024 // s, 1024 // s, 256]) for s in [4, 8, 16, 32, 64]]
    yx = tf.random.uniform([1, 4, 2], 0, 1) * 1024
    hw = tf.random.uniform([1, 4, 2]) * 1024
    boxes = tf.clip_by_value(tf.concat([yx - hw * 0.5, yx + hw * 0.5], -1), 0, 1024)

    pooled_features = pooling(features, boxes)
    # print(pooled_features)


if __name__ == '__main__':
    main()
