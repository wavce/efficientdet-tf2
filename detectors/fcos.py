import time
import tensorflow as tf

from heads import build_head 
from necks import build_neck
from detectors import Detector
from core.bbox import Distance2Box
from backbones import build_backbone


class FCOS(Detector):
    def __init__(self, cfg, **kwargs):
        self.distance2box = Distance2Box()
        if "head" in kwargs:
            head = kwargs.pop("head")
        else:
            head = build_head(cfg.head)
        super(FCOS, self).__init__(cfg, head, **kwargs)
     
    def build_model(self, training=True):
        inputs = tf.keras.layers.Input(shape=list(self.cfg.train.dataset.input_size) + [3])

        outputs = build_backbone(self.cfg.backbone.backbone,
                                 convolution=self.cfg.backbone.convolution,
                                 normalization=self.cfg.backbone.normalization.as_dict(),
                                 activation=self.cfg.backbone.activation,
                                 output_indices=self.cfg.backbone.output_indices,
                                 strides=self.cfg.backbone.strides,
                                 dilation_rates=self.cfg.backbone.dilation_rates,
                                 frozen_stages=self.cfg.backbone.frozen_stages,
                                 weight_decay=self.cfg.backbone.weight_decay,
                                 dropblock=self.cfg.backbone.dropblock,
                                 pretrained_weights_path=self.cfg.train.pretrained_weights_path,
                                 input_tensor=inputs,
                                 input_shape=self.cfg.train.dataset.input_size + [3]).build_model() 
        if self.cfg.neck is not None:
            for i in range(self.cfg.neck.repeats):
                outputs = build_neck(self.cfg.neck.neck,
                                     inputs=outputs,
                                     convolution=self.cfg.neck.convolution,
                                     normalization=self.cfg.neck.normalization.as_dict(),
                                     activation=self.cfg.neck.activation,
                                     feat_dims=self.cfg.neck.feat_dims,
                                     min_level=self.cfg.neck.min_level,
                                     max_level=self.cfg.neck.max_level,
                                     weight_decay=self.cfg.neck.weight_decay,
                                     dropblock=self.cfg.neck.dropblock,
                                     use_multiplication=self.cfg.neck.use_multiplication,
                                     add_extra_conv=False,
                                     name=self.cfg.neck.neck + str(i))

        outputs = self.head.build_head(outputs)

        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def summary_boxes(self, outputs, image_info):
        predicted_boxes, predicted_labels, predicted_centerness = outputs
        target_boxes = image_info["target_boxes"]
        target_labels = image_info["target_labels"]
        grid_y = image_info["grid_y"]
        grid_x = image_info["grid_x"]
        
        predicted_boxes = tf.cast(predicted_boxes, tf.float32)
        predicted_labels = tf.cast(predicted_labels, tf.float32)
        predicted_centerness = tf.cast(predicted_centerness, tf.float32)
        predicted_boxes = self.distance2box(predicted_boxes, grid_y, grid_x)
        matched_gt_boxes = self._get_matched_gt_boxes(target_boxes, target_labels >= 1)
        predicted_boxes = tf.clip_by_value(predicted_boxes, 0, 1)
       
        if self.cfg.head.use_sigmoid:
            predicted_scores = tf.nn.sigmoid(predicted_labels)
        else:
            predicted_scores = tf.nn.softmax(predicted_labels, axis=-1)
        
        predicted_scores = predicted_scores[:, :, 1:]  
        predicted_centerness = tf.nn.sigmoid(predicted_centerness)
        predicted_scores = predicted_scores * predicted_centerness      
        nmsed_boxes, nmsed_scores, _, _ = self.nms(predicted_boxes, predicted_scores)

        return matched_gt_boxes, nmsed_boxes, nmsed_scores

    
