import math
from configs import Config


def default_detection_configs(phi, 
                              min_level=3, 
                              max_level=7, 
                              fpn_filters=64,
                              neck_repeats=3,
                              head_repeats=3,
                              anchor_scale=4,
                              num_scales=3,
                              batch_size=4,
                              image_size=512,
                              fusion_type="weighted_sum"):
    h = Config()

    # model name
    h.detector = "efficientdet-d%d" % phi
    h.min_level = min_level
    h.max_level = max_level
    h.dtype = "float16"

    # backbone
    h.backbone = dict(backbone="efficientnet-b%d" % phi,
                      convolution="depthwise_conv2d",
                      dropblock=None,
                    #   dropblock=dict(keep_prob=None,
                    #                  block_size=None)
                      normalization=dict(normalization="batch_norm",
                                         momentum=0.99,
                                         epsilon=1e-3,
                                         axis=-1,
                                         trainable=False),
                      activation=dict(activation="swish"),
                      strides=[2, 1, 2, 2, 2, 1, 2, 1],
                      dilation_rates=[1, 1, 1, 1, 1, 1, 1, 1],
                      output_indices=[3, 4, 5],
                      frozen_stages=[-1])
    
    # neck
    h.neck = dict(neck="bifpn",
                  repeats=neck_repeats,
                  convolution="separable_conv2d",
                  dropblock=None,
                #   dropblock=dict(keep_prob=None,
                #                  block_size=None)
                  feat_dims=fpn_filters,
                  normalization=dict(normalization="batch_norm",
                                     momentum=0.99,
                                     epsilon=1e-3,
                                     axis=-1,
                                     trainable=False),
                  activation=dict(activation="swish"),
                  add_extra_conv=False,  # Add extra convolution for neck
                  fusion_type=fusion_type, 
                  use_multiplication=False)
    
    # head
    h.head = dict(head="RetinaNetHead",
                  repeats=head_repeats,
                  convolution="separable_conv2d",
                  dropblock=None,
                #   dropblock=dict(keep_prob=None,
                #                  block_size=None)
                  feat_dims=fpn_filters,
                  normalization=dict(normalization="batch_norm",
                                     momentum=0.99,
                                     epsilon=1e-3,
                                     axis=-1,
                                     trainable=False),
                  activation=dict(activation="swish"),
                  prior=0.01)
    
    # anchors parameters
    strides = [2 ** l for l in range(min_level, max_level + 1)]
    h.anchor = dict(aspect_ratios=[[1., 0.5, 2.]] * (max_level - min_level + 1),
                    scales=[
                        [2 ** (i / num_scales) * s * anchor_scale 
                        for i in range(num_scales)] for s in strides
                    ],
                    num_anchors=9)

    # assigner
    h.assigner = dict(assigner="max_iou_assigner",
                      pos_iou_thresh=0.5,
                      neg_iou_thresh=0.5)
    # sampler
    h.sampler = dict(sampler="pseudo_sampler")
    
    # loss
    h.use_sigmoid = True
    h.label_loss=dict(loss="focal_loss",
                      alpha=0.25,
                      gamma=1.5,
                      label_smoothing=0.,
                      weight=1.,
                      from_logits=True,
                      reduction="none")
    h.bbox_loss=dict(loss="smooth_l1_loss",
                     weight=50.,   # 50.
                     delta=.1,    # .1
                     reduction="none")
    # h.box_loss=dict(loss="giou_loss",
    #                 weight=10.,
    #                 reduction="none")
    h.weight_decay = 4e-5

    h.bbox_mean = None  # [0., 0., 0., 0.]
    h.bbox_std = None  # [0.1, 0.1, 0.2, 0.2]

    # dataset
    h.num_classes = 90
    h.skip_crowd_during_training = True
    h.dataset = "objects365"

    h.batch_size = batch_size
    h.input_size = [image_size, image_size]
    h.train_dataset_dir = "/home/bail/Data/data1/Dataset/Objects365/train"
    h.val_dataset_dir = "/home/bail/Data/data1/Dataset/Objects365/train"
    h.augmentation = [
        dict(ssd_crop=dict(patch_area_range=(0.3, 1.),
                            aspect_ratio_range=(0.5, 2.0),
                            min_overlaps=(0.1, 0.3, 0.5, 0.7, 0.9),
                            max_attempts=100,
                            probability=.5)),
        # dict(data_anchor_sampling=dict(anchor_scales=(16, 32, 64, 128, 256, 512),
        #                                overlap_threshold=0.7,
        #                                max_attempts=50,
        #                                probability=.5)),
        dict(flip_left_to_right=dict(probability=0.5)),
        dict(random_distort_color=dict(probability=1.))
        ]

    # train
    h.pretrained_weights_path = "/home/bail/Workspace/pretrained_weights/efficientdet-d%d" % phi

    h.optimizer = dict(optimizer="sgd", momentum=0.9)
    h.lookahead = None

    h.train_steps = 240000
    h.learning_rate_scheduler = dict(scheduler="cosine", initial_learning_rate=0.002)
    h.warmup = dict(warmup_learning_rate = 0.00001, steps = 24000)
    h.checkpoint_dir = "checkpoints/efficientdet_d%d" % phi
    h.summary_dir = "logs/efficientdet_d%d" % phi

    h.gradient_clip_norm = .0

    h.log_every_n_steps = 500
    h.save_ckpt_steps = 10000
    h.val_every_n_steps = 4000

    h.postprocess = dict(pre_nms_size=5000,   # select top_k high confident detections for nms 
                         post_nms_size=100,
                         iou_threshold=0.5,
                         score_threshold=0.2)
    
    return h


efficientdet_model_param_dict = {
    "efficientdet-d0": dict(phi=0, 
                            fpn_filters=64, 
                            neck_repeats=3, 
                            head_repeats=3, 
                            image_size=512),
    "efficientdet-d1": dict(phi=1, 
                            fpn_filters=88, 
                            neck_repeats=4, 
                            head_repeats=3, 
                            image_size=640),
    "efficientdet-d2": dict(phi=2, 
                            fpn_filters=112, 
                            neck_repeats=5, 
                            head_repeats=3, 
                            image_size=768),
    "efficientdet-d3": dict(phi=3, 
                            fpn_filters=160, 
                            neck_repeats=6, 
                            head_repeats=4, 
                            image_size=896),
    "efficientdet-d4": dict(phi=4, 
                            fpn_filters=224, 
                            neck_repeats=7, 
                            head_repeats=4, 
                            image_size=1024),
    "efficientdet-d5": dict(phi=5, 
                            fpn_filters=288, 
                            neck_repeats=7, 
                            head_repeats=4, 
                            image_size=1280),
    "efficientdet-d6": dict(phi=6, 
                            fpn_filters=384, 
                            neck_repeats=8, 
                            head_repeats=5, 
                            image_size=1280, 
                            fusion_type="sum"),
    "efficientdet-d7": dict(phi=7, 
                            fpn_filters=384, 
                            neck_repeats=8, 
                            head_repeats=5, 
                            anchor_scale=5.,
                            image_size=1536, 
                            fusion_type="sum"),
}


def get_efficientdet_config(model_name="efficientdet-d0"):
    return default_detection_configs(**efficientdet_model_param_dict[model_name])


if __name__ == "__main__":
    print(get_efficientdet_config("efficientdet-d7"))
