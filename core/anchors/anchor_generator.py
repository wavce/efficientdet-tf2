import math
import tensorflow as tf


class AnchorGenerator(object):
    def __init__(self, shifting=False, dtype=tf.float32):
        self.shifting = shifting
        self.dtype = dtype

    def __call__(self, feature_map_size, scales, aspect_ratios, strides):
        num_anchors = len(scales) * len(aspect_ratios)
        scales = tf.convert_to_tensor(scales, dtype=self.dtype)
        aspect_ratios = tf.convert_to_tensor(aspect_ratios, dtype=self.dtype)
        input_h = feature_map_size[0]
        input_w = feature_map_size[1]
        xx, yy = tf.meshgrid(tf.range(feature_map_size[0]),
                             tf.range(feature_map_size[1]))

        yy = (tf.cast(yy, self.dtype) + 0.5) * strides
        xx = (tf.cast(xx, self.dtype) + 0.5) * strides

        h_ratio = tf.math.sqrt(aspect_ratios)
        w_ratio = 1. / h_ratio

        ws = tf.reshape(scales[:, None] * w_ratio[None, :], [-1])
        hs = tf.reshape(scales[:, None] * h_ratio[None, :], [-1])

        yy = tf.expand_dims(yy, -1)
        xx = tf.expand_dims(xx, -1)
        anchors = tf.stack([yy - 0.5 * hs,
                            xx - 0.5 * ws,
                            yy + 0.5 * hs,
                            xx + 0.5 * ws], axis=-1)

        if self.shifting:
            # vertical
            anchors2 = tf.stack([yy + strides * 0.5 - 0.5 * hs,
                                 xx - 0.5 * ws,
                                 yy + strides * 0.5 + 0.5 * hs,
                                 xx + 0.5 * ws], axis=-1)
            # horizontal
            anchors3 = tf.stack([yy - 0.5 * hs,
                                 xx + strides * 0.5 - 0.5 * ws,
                                 yy + 0.5 * hs,
                                 xx + strides * 0.5 + 0.5 * ws], axis=-1)
            # diagonal
            shifting_strides = strides / math.sqrt(2)
            anchors4 = tf.stack([yy + shifting_strides - 0.5 * hs,
                                 xx + shifting_strides - 0.5 * ws,
                                 yy + shifting_strides + 0.5 * hs,
                                 xx + shifting_strides + 0.5 * ws], axis=-1)

            anchors = tf.concat([anchors, anchors2, anchors3, anchors4], axis=0)

        anchors = tf.reshape(anchors, [input_h * input_w * num_anchors, 4])
        anchors /= tf.convert_to_tensor([input_h * strides, input_w * strides, 
                                         input_h * strides, input_w * strides], tf.float32)

        return anchors

