from abc import ABCMeta
from abc import abstractmethod
import tensorflow as tf

from core.losses import build_loss
from core.samplers import build_sampler


class Head(object):
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self._use_iou_loss = "iou" in cfg.bbox_loss.loss
        self.bbox_loss_func = build_loss(**cfg.bbox_loss.as_dict())
        self.label_loss_func = build_loss(**cfg.label_loss.as_dict())

        self.sampler = build_sampler(**cfg.sampler.as_dict())

    @property
    def min_level(self):
        return self.cfg.min_level
    
    @property
    def max_level(self):
        return self.cfg.max_level
    
    @property
    def image_data_format(self):
        return tf.keras.backend.image_data_format()

    @property
    def num_classes(self):
        return self.cfg.num_classes

    @abstractmethod
    def build_head(self, inputs):
        pass
    
    def _compute_losses_per_image(self, target_boxes, target_labels, predicted_boxes, predicted_labels, box_weights, label_weights, num_pos):
        with tf.name_scope("compute_losses_per_image"):
            if self.cfg.use_sigmoid:
                target_labels -= 1
            target_labels = tf.one_hot(tf.cast(target_labels, tf.int32), self.num_classes)
            label_loss = self.label_loss_func(target_labels, predicted_labels, label_weights)
            bbox_loss = self.bbox_loss_func(target_boxes, predicted_boxes, box_weights)
            label_loss = tf.reduce_sum(label_loss) * (1. / tf.cast(num_pos + 1, label_loss.dtype))
            bbox_loss = tf.reduce_sum(bbox_loss) *  (1. / tf.cast(num_pos + 1, label_loss.dtype))

            return bbox_loss, label_loss

    def compute_losses(self, outputs, image_info):
        with tf.name_scope("compute_losses"):
            box_loss_ta = tf.TensorArray(size=1, dynamic_size=True, dtype=tf.float32)
            label_loss_ta = tf.TensorArray(size=1, dynamic_size=True, dtype=tf.float32)

            predicted_boxes, predicted_labels = outputs["predicted_boxes"], outputs["predicted_labels"]
            predicted_boxes = tf.cast(predicted_boxes, tf.float32)
            predicted_labels = tf.cast(predicted_labels, tf.float32)
            target_boxes = image_info["target_boxes"]
            target_labels = image_info["target_labels"]
            total_anchors = image_info["total_anchors"]

            for i in tf.range(tf.shape(target_boxes)[0]):
                sampling_result = self.sampler.sample(target_boxes[i], target_labels[i])
                t_boxes = sampling_result["target_boxes"]
                p_boxes = predicted_boxes[i]
                if self._use_iou_loss:
                    p_boxes = self.delta2box(total_anchors[i], p_boxes)
                else:
                    t_boxes = self.box2delta(total_anchors[i], t_boxes)
                b_loss, l_loss = self._compute_losses_per_image(
                    t_boxes, sampling_result["target_labels"], p_boxes, predicted_labels[i], 
                    sampling_result["box_weights"], sampling_result["label_weights"],
                    sampling_result["num_pos"]) 
                
                box_loss_ta = box_loss_ta.write(i, b_loss)
                label_loss_ta = label_loss_ta.write(i, l_loss)
            
            box_loss = tf.reduce_mean(box_loss_ta.stack())
            label_loss = tf.reduce_mean(label_loss_ta.stack())

            return dict(box_loss=box_loss, label_loss=label_loss)
