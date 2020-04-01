import time
import tensorflow as tf
from detectors import FCOS
from necks import build_neck
from heads import build_head
from backbones import build_backbone


class EfficientFCOS(FCOS):
    def __init__(self, cfg, **kwargs):
        head = build_head(cfg)

        super(EfficientFCOS, self).__init__(cfg, head=head, **kwargs)

    def build_model(self, training=True):
        inputs = tf.keras.Input(self.cfg.train.dataset.input_size + [3], name="inputs")
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
                                 add_extra_conv=self.cfg.neck.add_extra_conv,
                                 dropblock=self.cfg.neck.dropblock,
                                 name=self.cfg.neck.neck + "_" + str(i))

        outputs = self.head.build_head(outputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name=self.cfg.detector)

   