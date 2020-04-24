import time
import tensorflow as tf
from necks import build_neck
from heads import build_head 
from detectors import Detector
from core.bbox import Delta2Box
from backbones import build_backbone


class EfficientDet(Detector):
    def __init__(self, cfg, **kwargs):
        head = build_head(cfg)

        super(EfficientDet, self).__init__(cfg, head, **kwargs)

        self.delta2box = Delta2Box(mean=cfg.bbox_mean, std=cfg.bbox_std)
                                   
    def build_model(self):
        inputs = tf.keras.Input(list(self.cfg.input_size) + [3], name="inputs")
        outputs = build_backbone(self.cfg.backbone.backbone,
                                 convolution=self.cfg.backbone.convolution,
                                 normalization=self.cfg.backbone.normalization.as_dict(),
                                 activation=self.cfg.backbone.activation,
                                 output_indices=self.cfg.backbone.output_indices,
                                 strides=self.cfg.backbone.strides,
                                 dilation_rates=self.cfg.backbone.dilation_rates,
                                 frozen_stages=self.cfg.backbone.frozen_stages,
                                 dropblock=self.cfg.backbone.dropblock,
                                 input_tensor=inputs,
                                 input_shape=self.cfg.input_size + [3]).build_model()
    
        outputs = build_neck(self.cfg.neck.neck,
                             repeats=self.cfg.neck.repeats,
                             inputs=outputs,
                             convolution=self.cfg.neck.convolution,
                             normalization=self.cfg.neck.normalization.as_dict(),
                             activation=self.cfg.neck.activation.as_dict(),
                             feat_dims=self.cfg.neck.feat_dims,
                             min_level=self.cfg.min_level,
                             max_level=self.cfg.max_level,
                             add_extra_conv=self.cfg.neck.add_extra_conv,
                             dropblock=self.cfg.neck.dropblock,
                             fusion_type=self.cfg.neck.fusion_type,
                             input_size=self.cfg.input_size) 

        outputs = self.head.build_head(outputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name=self.cfg.detector)
    
    def init_weights(self, pretrained_weight_path=None):
        if pretrained_weight_path is not None:
            pretrained_weights = tf.train.latest_checkpoint(pretrained_weight_path)
            use_exponential_moving_average = False
            # for w in tf.train.list_variables(pretrained_weights):
            #     if "ExponentialMovingAverage" not in w[0]:
            #         # use_exponential_moving_average = True
            #         print(w[0], w[1])

            for weight in self.model.weights:
                name = weight.name.split(":")[0]
                # print(name, weight.shape)
                # if "box-predict" in name or "class-predict" in name:
                #     continue
                if "batch_normalization" in name:
                    name = name.replace("batch_normalization", "tpu_batch_normalization")
                # if use_exponential_moving_average:
                #     name += "/ExponentialMovingAverage"
                try:
                    pretrained_weight = tf.train.load_variable(pretrained_weights, name)
                    weight.assign(pretrained_weight)
                except:
                    print("{} not in {}.".format(name, pretrained_weight_path))


