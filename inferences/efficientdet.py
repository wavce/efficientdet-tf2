import argparse
import numpy as  np
import tensorflow as tf
from core.bbox import Delta2Box
from configs import build_configs
from core.layers import build_nms
from detectors import build_detector
from core.anchors import AnchorGenerator


def _generate_anchor_configs(min_level, max_level, num_scales, aspect_ratios):
    """Generates mapping from output level to a list of anchor configurations.
        A configuration is a tuple of (num_anchors, scale, aspect_ratio).
        Args:
            min_level: integer number of minimum level of the output feature pyramid.
            max_level: integer number of maximum level of the output feature pyramid.
            num_scales: integer number representing intermediate scales added
                on each level. For instances, num_scales=2 adds two additional
                anchor scales [2^0, 2^0.5] on each level.
            aspect_ratios: list of tuples representing the aspect ratio anchors added
                on each level. For instances, aspect_ratios =
                [(1, 1), (1.4, 0.7), (0.7, 1.4)] adds three anchors on each level.
        Returns:
            anchor_configs: a dictionary with keys as the levels of anchors and
            values as a list of anchor configuration.
    """
    anchor_configs = {}
    for level in range(min_level, max_level + 1):
        anchor_configs[level] = []
        for scale_octave in range(num_scales):
            for aspect in aspect_ratios:
                anchor_configs[level].append((2 ** level, scale_octave / float(num_scales), aspect))
    return anchor_configs


def _generate_anchor_boxes(image_size, anchor_scale, anchor_configs):
    """Generates multiscale anchor boxes.
        Args:
            image_size: integer number of input image size. The input image has the
            same dimension for width and height. The image_size should be divided by
            the largest feature stride 2^max_level.
            anchor_scale: float number representing the scale of size of the base
            anchor to the feature stride 2^level.
            anchor_configs: a dictionary with keys as the levels of anchors and
            values as a list of anchor configuration.
        Returns:
            anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all
            feature levels.
        Raises:
            ValueError: input size must be the multiple of largest feature stride.
    """
    boxes_all = []
    for _, configs in anchor_configs.items():
        boxes_level = []
        for config in configs:
            stride, octave_scale, aspect = config
            if image_size[0] % stride != 0:
                raise ValueError('input size must be divided by the stride.')
            base_anchor_size = anchor_scale * stride * 2**octave_scale
            anchor_size_x_2 = base_anchor_size * aspect[0] / 2.0
            anchor_size_y_2 = base_anchor_size * aspect[1] / 2.0

            x = np.arange(stride / 2, image_size[1], stride)
            y = np.arange(stride / 2, image_size[0], stride)
            xv, yv = np.meshgrid(x, y)
            xv = xv.reshape(-1)
            yv = yv.reshape(-1)

            boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                               yv + anchor_size_y_2, xv + anchor_size_x_2))
            boxes = np.swapaxes(boxes, 0, 1)
            boxes_level.append(np.expand_dims(boxes, axis=1))
        # concat anchors on the same level to the reshape NxAx4
        boxes_level = np.concatenate(boxes_level, axis=1)
        boxes_all.append(boxes_level.reshape([-1, 4]))

    anchor_boxes = np.vstack(boxes_all)
    return anchor_boxes


class EfficientDet(tf.keras.Model):
    def __init__(self, model_name="efficientdet-d0", image_size=None, **kwargs):
        super(EfficientDet, self).__init__(**kwargs)
        cfg = build_configs(model_name)

        self.input_size = cfg.input_size if image_size is None else image_size
        cfg.override(dict(input_size=self.input_size))
       
        self.model = build_detector(cfg.detector, cfg=cfg).model
        
        self.nms = build_nms("combined_non_max_suppression", 
                             pre_nms_size=cfg.postprocess.pre_nms_size,
                             post_nms_size=cfg.postprocess.post_nms_size,
                             iou_threshold=cfg.postprocess.iou_threshold,
                             score_threshold=cfg.postprocess.score_threshold,
                             num_classes=cfg.num_classes)
        self.delta2box = Delta2Box(mean=cfg.bbox_mean, std=cfg.bbox_std)
        self.aspect_ratios = cfg.anchor.aspect_ratios
        self.anchor_scales = cfg.anchor.scales
        # base_scale = cfg.anchor_scale
        # strides = [8, 16, 32, 64, 128]
        # self.anchor_scales = [[2 ** (i / 3) * s * base_scale for i in range(3)] for s in strides]
        # anchors_configs = _generate_anchor_configs(3, 7, 3, [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        # anchors = _generate_anchor_boxes(self.input_size, 4, anchors_configs)
 
        # self.anchors = tf.convert_to_tensor([anchors], tf.float32)
        # self.normalizer = tf.convert_to_tensor(
        #     [[[self.input_size[0], self.input_size[1], self.input_size[0], self.input_size[1]]]], tf.float32)
        self.anchor_generator = AnchorGenerator()

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs):
        inputs = tf.image.convert_image_dtype(inputs, tf.float32)
        total_anchors = []
        input_size = tf.shape(inputs)[1:3]

        normalizer = tf.convert_to_tensor(
            [[[input_size[0], input_size[1], input_size[0], input_size[1]]]], tf.float32)
        for i, level in enumerate(range(3, 7 + 1)):
            anchors = self.anchor_generator(
                input_size // (2 ** level), self.anchor_scales[i], self.aspect_ratios[i], 2 ** level)
            total_anchors.append(anchors)
        total_anchors = tf.concat(total_anchors, 0)
        outputs = self.model(inputs, training=False)
        predicted_boxes, predicted_labels = outputs["predicted_boxes"], outputs["predicted_labels"]
        predicted_boxes = self.delta2box(total_anchors, predicted_boxes)
        predicted_boxes = tf.clip_by_value(predicted_boxes / normalizer, 0, 1)
        predicted_scores = tf.nn.sigmoid(predicted_labels)
        # tf.print(predicted_boxes)
        # tf.print(tf.reduce_max(predicted_scores))

        return self.nms(predicted_boxes, predicted_scores)


def save_model(model_name="efficientdet-d0", image_size=None):
    efficientdet = EfficientDet(model_name, image_size)
    input_size = efficientdet.input_size
    efficientdet(tf.cast(tf.random.uniform([1] + list(input_size) + [3], 0, 256), tf.uint8), training=False)
    # test(efficientdet)
    tf.saved_model.save(efficientdet, "./saved_model/{}/1/".format(model_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sabed model args")
    parser.add_argument("--model_name", default="efficientdet-d0", type=str)
    parser.add_argument("--input_size", default=None, type=str)

    args = parser.parse_args()
    
    input_size = args.input_size
    if input_size is not None:
    	assert "x" in input_size, "input_size must like 512x512"
    	input_size = [int(s) for s in input_size.split("x")]

    save_model(args.model_name, input_size)

