import math
import tensorflow as tf
from heads import Head
from core.bbox import Box2Delta
from core.bbox import Delta2Box
from heads import prediction_head
from core.losses import build_loss
from core.samplers import build_sampler
from core.layers import build_convolution


class RetinaNetHead(Head):
    def __init__(self, cfg, **kwargs):
        super(RetinaNetHead, self).__init__(cfg, **kwargs)
        
        self.box_shared_convs = [
            build_convolution(convolution=cfg.head.convolution,
                              filters=cfg.head.feat_dims,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              padding="same",
                              use_bias=cfg.head.normalization is None or cfg.head.convolution == "separable_conv2d",
                              kernel_regularizer=self.kernel_regularizer,
                              name="box_net/box-%d" % i)
            for i in range(cfg.head.repeats)
        ]
        self.class_shared_convs = [
            build_convolution(convolution=cfg.head.convolution,
                             filters=cfg.head.feat_dims,
                             kernel_size=(3, 3),
                             strides=(1, 1),
                             padding="same",
                             use_bias=cfg.head.normalization is None or cfg.head.convolution == "separable_conv2d",
                             kernel_regularizer=self.kernel_regularizer,
                             name="class_net/class-%d" % i)
            for i in range(cfg.head.repeats)
        ]

        self.classifier = build_convolution(cfg.head.convolution,
                                            filters=cfg.head.num_classes * cfg.head.num_anchors,
                                            kernel_size=(3, 3),
                                            strides=(1, 1),
                                            padding="same",
                                            use_bias=True,
                                            bias_initializer=tf.keras.initializers.Constant(
                                                -math.log((1. - cfg.head.prior) / cfg.head.prior)),
                                            name="class_net/class-predict")
        self.regressor = build_convolution(cfg.head.convolution,
                                           filters=4 * cfg.head.num_anchors,
                                           kernel_size=(3, 3),
                                           strides=(1, 1),
                                           padding="same",
                                           use_bias=True,
                                           name="box_net/box-predict")
        self.delta2box = Delta2Box(mean=cfg.bbox_decoder.bbox_mean,
                                   std=cfg.bbox_decoder.bbox_std)
        self.box2delta = Box2Delta(mean=cfg.bbox_decoder.bbox_mean,
                                   std=cfg.bbox_decoder.bbox_std)

        self._use_iou_loss = "iou" in cfg.loss.bbox_loss.loss
        self.bbox_loss_func = build_loss(**cfg.loss.bbox_loss.as_dict())
        self.label_loss_func = build_loss(**cfg.loss.label_loss.as_dict())
        self.sampler = build_sampler(**cfg.sampler.as_dict())
        self.cfg = cfg

    def build_head(self, inputs):
        predicted_boxes = list()
        predicted_labels = list()
        for i, level in enumerate(range(self.cfg.head.min_level, self.cfg.head.max_level+1)):
            box_feat = prediction_head(inputs[i],
                                       self.box_shared_convs,
                                       normalization=self.cfg.head.normalization,
                                       activation=self.cfg.head.activation,
                                       repeats=self.cfg.head.repeats,
                                       level=level,
                                       name="box_net/box")
            p_boxes = self.regressor(box_feat)
            label_feat = prediction_head(inputs[i],
                                         self.class_shared_convs,
                                         normalization=self.cfg.head.normalization,
                                         activation=self.cfg.head.activation,
                                         repeats=self.cfg.head.repeats,
                                         level=level,
                                         name="class_net/class")
            p_labels = self.classifier(label_feat)

            if tf.keras.backend.image_data_format() == "channels_first":
                p_boxes = tf.keras.layers.Permute(
                    (2, 3, 1), name="box_net/permute_%d" % level)(p_boxes)
                p_labels = tf.keras.layers.Permute(
                    (2, 3, 1), name="class_net/permute_%d" % level)(p_labels)
            p_boxes = tf.keras.layers.Reshape(
                [-1, 4], name="box_net/reshape_%d" % level)(p_boxes)
            p_labels = tf.keras.layers.Reshape(
                [-1, self.num_classes], name="class_net/reshape_%d" % level)(p_labels)
           
            predicted_boxes.append(p_boxes)
            predicted_labels.append(p_labels)
        
        predicted_boxes = tf.keras.layers.Concatenate(
            axis=1, name="box_net/concat")(predicted_boxes)
        predicted_labels = tf.keras.layers.Concatenate(
            axis=1, name="class_net/concat")(predicted_labels)

        return predicted_boxes, predicted_labels
    