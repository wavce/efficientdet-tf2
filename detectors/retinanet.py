import time
import tensorflow as tf
from necks import build_neck
from heads import build_head 
from detectors import Detector
from core.bbox import Delta2Box


class RetinaNet(Detector):
    def __init__(self, cfg, **kwargs):
        head = build_head(cfg)

        super(RetinaNet, self).__init__(cfg, head, **kwargs)

        self.delta2box = Delta2Box(mean=cfg.bbox_decoder.bbox_mean,
                                   std=cfg.bbox_decoder.bbox_std)
                                   
    def build_model(self, training=True):
        inputs = tf.keras.Input(list(self.cfg.train.dataset.input_size) + [3], name="inputs")
        outputs = self.backbone(inputs)

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