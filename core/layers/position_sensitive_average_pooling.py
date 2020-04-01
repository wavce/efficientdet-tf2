import tensorflow as tf


class PSAvgPooling(tf.keras.layers.Layer):
    def __init__(self, num_boxes, crop_size=(9, 9), num_spatial_bins=(3, 3), **kwargs):
        super(PSAvgPooling, self).__init__(**kwargs)

        total_bins = 1
        bin_crop_size = []
        for num_bins, crop_dim in zip(num_spatial_bins, crop_size):
            if num_bins < 1:
                raise ValueError("num_spatial_bins should be >= 1.")

            if crop_dim % num_bins != 0:
                raise ValueError("crop_size should be divisible by num_spatial_bins.")

            total_bins *= num_bins
            bin_crop_size.append(crop_dim // num_bins)

        if bin_crop_size[0] != bin_crop_size[1]:
            raise ValueError("Only support square bin crop size for now.")

        self.num_spatial_bins = num_spatial_bins
        self.bin_crop_size = bin_crop_size
        self.total_bins = total_bins
        self.num_boxes = num_boxes

    def build(self, input_shape):
        self.weights = self.add_weight(name="weight",
                                       shape=[1, 1, 1, 1, self.num_spatial_bins],
                                       dtype=self.dtype,
                                       initializer=tf.keras.initializers.Ones())

    def _one_image_pooling(self, image, boxes):
        y1, x1, y2, x2 = tf.unstack(boxes, 4, -1)
        ps_boxes = tf.TensorArray(size=self.total_bins, dtype=boxes.dtype)

        i = tf.constant(0, tf.int32)
        for bin_y in tf.range(self.num_spatial_bins[0], dtype=boxes.dtype):
            step_y = (y2 - y1) / tf.cast(self.num_spatial_bins[0], dtype=boxes.dtype)
            for bin_x in tf.range(self.num_spatial_bins[1], dtype=boxes.dtype):
                step_x = (x2 - x1) / tf.cast(self.num_spatial_bins[1], dtype=boxes.dtype)
                i += 1
                box = tf.convert_to_tensor([y1 + bin_y * step_y,
                                            x1 + bin_x * step_x,
                                            y1 + bin_y * (step_y + 1),
                                            x1 + bin_x * (step_x + 1)], dtype=boxes.dtype)
                ps_boxes.write(i, box)

        step_split = tf.shape(image)[-1] // self.total_bins
        image_crops = tf.TensorArray(size=self.total_bins, dtype=image.dtype)
        # image_splits = tf.split(image, self.total_bins, axis=-1)
        for i in tf.range(self.total_bins):
            split = image[..., i*step_split: (i+1)*step_split]
            split_crop = tf.image.crop_and_resize(image=tf.expand_dims(split, 0),
                                                  boxes=ps_boxes.read(i),
                                                  box_indices=tf.zeros(tf.shape(boxes)[0], dtype=tf.int32),
                                                  crop_size=self.bin_crop_size)
            image_crops.write(i, split_crop)

        features = image_crops.stack(axis=-1)  # [num_boxes, crop_height, crop_width, depth, num_bins]
        features = tf.reduce_mean(features, [1, 2], keepdims=True)  # [num_boxes, 1, 1, depth, num_bins]

        features = tf.reduce_mean(features * self.weights, -1)  # [num_boxes, 1, 1, depth]

        ps_boxes.close()
        image_crops.close()

        return features

    def call(self, inputs, boxes):
        batch_size = tf.shape(inputs)[0]
        features = tf.TensorArray(size=batch_size)

        for i in tf.range(batch_size):
            feat = self._one_image_pooling(inputs[0], boxes[0])
            features.write(i, feat)

        return features.concate(axis=0)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([self.num_boxes, 1, 1, input_shape[-1] // self.total_bins])

    def get_config(self,):
        config = {"num_boxes": self.num_boxes,
                  "bin_crop_size": self.bin_crop_size,
                  "num_spatial_bins": self.num_spatial_bins}

        base_config = super(PSAvgPooling, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
