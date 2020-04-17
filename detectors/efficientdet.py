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

        self.delta2box = Delta2Box(mean=cfg.bbox_decoder.bbox_mean,
                                   std=cfg.bbox_decoder.bbox_std)
                                   
    def build_model(self):
        inputs = tf.keras.Input(list(self.cfg.train.dataset.input_size) + [3], name="inputs")
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
                                 input_tensor=inputs,
                                 input_shape=self.cfg.train.dataset.input_size + [3]).build_model()
      
        outputs = build_neck(self.cfg.neck.neck,
                             repeats=self.cfg.neck.repeats,
                             inputs=outputs,
                             convolution=self.cfg.neck.convolution,
                             normalization=self.cfg.neck.normalization.as_dict(),
                             activation=self.cfg.neck.activation.as_dict(),
                             feat_dims=self.cfg.neck.feat_dims,
                             min_level=self.cfg.neck.min_level,
                             max_level=self.cfg.neck.max_level,
                             weight_decay=self.cfg.neck.weight_decay,
                             add_extra_conv=self.cfg.neck.add_extra_conv,
                             dropblock=self.cfg.neck.dropblock,
                             input_size=self.cfg.train.dataset.input_size[0]) 
       
        outputs = self.head.build_head(outputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name=self.cfg.detector)
    
    def init_weights(self, pretrained_weight_path=None):
        if pretrained_weight_path is not None:
            pretrained_weights = tf.train.latest_checkpoint(pretrained_weight_path)
            use_exponential_moving_average = False
            # for w in tf.train.list_variables(pretrained_weights):
            #     if "ExponentialMovingAverage" in w[0]:
            #         # use_exponential_moving_average = True
            #         print(w[0], w[1])

            for weight in self.model.weights:
                name = weight.name.split(":")[0]
                # if "/se/" in name and "efficientnet" in name:
                # print(name, weight.shape)
                # if "box_net" in name or "class_net" in name:
                #     print(name, weight.shape)
                if "batch_normalization" in name:
                    name = name.replace("batch_normalization", "tpu_batch_normalization")
                # print(name, weight.shape)
                # if use_exponential_moving_average:
                #     name += "/ExponentialMovingAverage"
                try:
                    pretrained_weight = tf.train.load_variable(pretrained_weights, name)
                    weight.assign(pretrained_weight)
                except:
                    print("{} not in {}.".format(name, pretrained_weight_path))

            tf.print("Restored pre-trained weights from %s" % pretrained_weights)


