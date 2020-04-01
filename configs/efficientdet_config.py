import math
from configs import params_dict


def fpn_filters(phi, upper=6):
    if phi >= upper:
        phi = upper

    multiplier = math.ceil(64 * (1.35 ** phi) / 8)
    if phi == 4:
        multiplier += 1

    filters = multiplier * 8

    return int(filters)


def num_heads(phi, upper=6):
    if phi >= upper:
        phi = upper

    return int(3 + math.floor(phi / 3))


num_classes = 90

min_level = 3
max_level = 7
phi = 0
batch_size = 2
# num_scales = 3
# base_scale = 4
# strides = [8, 16, 32, 64, 128]
# anchor_scales = [[2 ** (i / num_scales) * s * base_scale
#                   for i in range(num_scales)] for s in strides]
num_scales = 3 # 3
aspect_ratios = [1., 0.5, 2.]
base_scale = 8  # 8
step = 2 ** (1 / (num_scales - 1))
strides = [8, 16, 32, 64, 128]
anchor_scales = [[base_scale * (step ** i) for i in range(0, num_scales)],
                 [base_scale * (step ** i) for i in range(num_scales, num_scales * 2)],
                 [base_scale * (step ** i) for i in range(num_scales * 2, num_scales * 3)],
                 [base_scale * (step ** i) for i in range(num_scales * 3, num_scales * 4)],
                 [base_scale * (step ** i) for i in range(num_scales * 4, num_scales * 5)]]
anchor_scales = [[8.0, 12., 16.], [22., 32., 45.], [64., 90., 128.], [181., 256., 362.], [406., 448., 512.]]  # input=512
# anchor_scales = [[8.0, 12., 16.], [22., 32., 45.], [64., 90., 128.], [181., 256., 362.], [406., 512., 640.]]  # input=640
# anchor_scales = [[8.0, 12., 16.], [22., 32., 45.], [64., 90., 128.], [181., 256., 362.], [512., 640., 768.]]  # input=768
# anchor_scales = [[8.0, 12., 16.], [22., 32., 45.], [64., 90., 128.], [181., 256., 362.], [512., 640., 986.]]  # input=896

input_size = int(512 + phi * 128)
CFG = params_dict.ParamsDict(default_params={
    "detector": "efficientdet",
    "dtype": "float32",  # model dtype, if float16, means use mixed precision training.
    "phi": phi,
    "backbone": {
        "backbone": "efficientnet-b%d" % phi,
        "convolution": None,
        # "dropblock": {
        #     "dropblock_keep_prob": None,
        #     "dropblock_size": None,
        # },
        "dropblock": None,
        "normalization": {
            "normalization": "batch_norm",
            "momentum": 0.997,
            "epsilon": 1e-4,
            "axis": -1,
            "trainable": False
        },
        "activation": {"activation": "swish"},
        "strides": [2, 1, 2, 2, 2, 1, 2, 1],
        "dilation_rates": [1, 1, 1, 1, 1, 1, 1, 1],
        "output_indices": [3, 4, 5],
        "frozen_stages": [-1],
        "weight_decay": 4e-5
    },
    "neck": {
        "neck": "bifpn",
        "repeats": int(phi + 3),
        "convolution": "separable_conv2d",
        "feat_dims": fpn_filters(phi),
        # "normalization":  {
        #     "normalization": "filter_response_norm",
        # },
        # "activation": None,
        "normalization": {
            "normalization": "batch_norm",
            "momentum": 0.997,
            "epsilon": 1e-4,
            "axis": -1,
            "trainable": True
        },
        "activation": {"activation": "swish"},
        "dropblock": None,
        "add_extra_conv": False,  # Add extra convolution for neck
        "use_multiplication": False,  # Use multiplication in neck, default False
        "min_level": min_level,
        "max_level": max_level,
        "weight_decay": 4e-5
    },
    "head": {
        "head": "RetinaNetHead",
        "repeats": num_heads(phi),
        "convolution": "separable_conv2d",
        # "normalization":  {
        #     "normalization": "filter_response_norm",
        # },
        # "activation": None,
        "normalization": {
            "normalization": "batch_norm",
            "momentum": 0.997,
            "epsilon": 1e-4,
            "axis": -1,
            "trainable": True
        },
        "activation": {"activation": "swish"},
        "dropblock": None,
        "feat_dims": fpn_filters(phi),
        "num_anchors": num_scales  * len(aspect_ratios),
        "num_classes": num_classes,  # 2
        "strides": strides,
        "prior": 0.01,
        "weight_decay": 4e-5,
        "use_sigmoid": True,
        "min_level": min_level,
        "max_level": max_level
    },
    "anchor": {
        "num_anchors": num_scales * len(aspect_ratios),
        "strides": strides,
        "scales": anchor_scales,
        "aspect_ratios": [aspect_ratios] * (max_level - min_level + 1),
        "min_level": min_level,
        "max_level": max_level,
    },
    "assigner": {
        "assigner": "max_iou_assigner",
        "pos_iou_thresh": 0.35,
        "neg_iou_thresh": 0.35,
        "min_level": min_level,
        "max_level": max_level
    },
    "sampler": {"sampler": "pseudo_sampler"},
    "loss": {
        "label_loss": {
            "loss": "focal_loss",
            "alpha": 0.25,
            "gamma": 1.5,
            "label_smoothing": 0.,
            "weight": 1.,
            "from_logits": True,
            "use_sigmoid": True,
            "reduction": "none"
        },
        "bbox_loss": {
            "loss": "smooth_l1_loss",
            "weight": 50.,   # 50.
            "delta": .1,    # .1
            "reduction": "none"
        },
        # "bbox_loss": {
        #     "loss": "giou_loss",
        #     "weight": 10.,
        #     "reduction": "none"
        # },
        "weight_decay": 4e-5
    },
    "bbox_decoder": {
            "bbox_mean": [0., 0., 0., 0.],
            "bbox_std": [0.1, 0.1, 0.2, 0.2]
        },
    "bbox_encoder":  {
            "bbox_mean": [0., 0., 0., 0.],
            "bbox_std": [0.1, 0.1, 0.2, 0.2]
        },
       "train": {
        "dataset": {
            "dataset": "objects365",
            "batch_size": batch_size,
            "input_size": [input_size, input_size],
            "dataset_dir": "/home/bail/Data/data1/Dataset/Objects365/train",
            "training": True,
            "augmentation": [
                dict(ssd_crop=dict(input_size=[input_size, input_size],
                                   patch_area_range=(0.3, 1.),
                                   aspect_ratio_range=(0.5, 2.0),
                                   min_overlaps=(0.1, 0.3, 0.5, 0.7, 0.9),
                                   max_attempts=100,
                                   probability=.5)),
                # dict(data_anchor_sampling=dict(input_size=[input_size, input_size],
                #                                anchor_scales=(16, 32, 64, 128, 256, 512),
                #                                overlap_threshold=0.7,
                #                                max_attempts=50,
                #                                probability=.5)),
                dict(flip_left_to_right=dict(probability=0.5)),
                dict(random_distort_color=dict(probability=1.))
            ]
        },
        "samples": 12876,
        "num_classes": num_classes,  # 2 

        "pretrained_weights_path": "/home/bail/Workspace/pretrained_weights/efficientdet-d%d" % phi,

        "optimizer": {
            "optimizer": "sgd",
            "momentum": 0.9,
        },
        "lookahead": None,
        "mixed_precision": {
            "loss_scale": None,  # The loss scale in mixed precision training. If None, use dynamic.
        },

        "train_steps": 240000,
        "learning_rate_scheduler": {
            # "learning_rate_scheduler": "piecewise_constant",
            # "initial_learning_rate": initial_learning_rate,
            # "boundaries": boundaries,
            # "values": values
            "learning_rate_scheduler": "cosine",
            "initial_learning_rate": 0.002
        },
        "warmup": {
            "warmup_learning_rate": 0.00001,
            "steps": 24000,
        },
        "checkpoint_dir": "checkpoints/efficientdet_d%d" % phi,
        "summary_dir": "logs/efficientdet_d%d" % phi,

        "gradient_clip_norm": .0,

        "log_every_n_steps": 500,
        "save_ckpt_steps": 10000,
    },
    "val": {
        "dataset": {
            "dataset": "objects365",
            "batch_size": batch_size,
            "input_size": [input_size, input_size],
            "dataset_dir": "/home/bail/Data/data1/Dataset/Objects365/train",
            "training": False,
            "augmentation": None,
        },
        "samples": 3222,
        "num_classes": num_classes,
        "val_every_n_steps": 15000,
    }, 
    "postprocess": {
        "pre_nms_size": 200,   # select top_k high confident detections for nms 
        "post_nms_size": 50,
        "iou_threshold": 0.5,
        "score_threshold": 0.5,
        "use_sigmoid": True,
    }},
    restrictions=[
        "head.num_classes == train.num_classes",
        "neck.feat_dims == head.feat_dims",
        "bbox_decoder.bbox_mean == bbox_encoder.bbox_mean",
        "bbox_decoder.bbox_std == bbox_encoder.bbox_std",
        "loss.weight_decay == head.weight_decay",
        "loss.weight_decay == neck.weight_decay",
        "train.dataset.dataset == val.dataset.dataset"
])


if __name__ == "__main__":
    # print(CFG.as_dict())
    # print(CFG.train.learning_rate_scheduler.as_dict())
    print(anchor_scales)
