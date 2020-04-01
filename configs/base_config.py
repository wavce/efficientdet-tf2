BASE_CFG = {
    "model_dir": "",
    "dtype": "float32",
    "backbone": {
        "name": "resnet50",
        "convolution": "conv2d",
        "dropblock": None,  # {
            # "dropblock_keep_prob": None,
            # "dropblock_size": None,
        # },
        "normalization": {
            "name": "batch_norm",
            "momentum": 0.997,
            "epsilon": 1e-4,
            "axis": -1,
            "trainable": True
        },
        "activation": "relu",
        "strides": [2, 2, 2, 2, 2],
        "dilation_rates": [1, 1, 1, 1, 1],
        "output_indices": [2, 3, 4],
        "frozen_stages": [0, 1],
    },
    "neck": {
        "convolution": "conv2d",
        "dropblock": {
            "dropblock_keep_prob": None,
            "dropblock_size": None,
        },
        "normalization": {
            "name": "batch_norm",
            "momentum": 0.997,
            "epsilon": 1e-4,
            "axis": -1,
            "trainable": True
        },
        "activation": "relu",
        "feat_dims": 256,
    },
    "head": {
        "name": "",
        "convolution": "conv2d",
        "dropblock": {
            "dropblock_keep_prob": None,
            "dropblock_size": None,
        },
        "normalization": {
            "name": "batch_norm",
            "momentum": 0.997,
            "epsilon": 1e-4,
            "axis": -1,
            "trainable": True
        },
        "activation": "relu",
        "feat_dims": 256,
        "num_anchors": 9,
        "num_classes": 91,
        "anchor_strides": [4, 8, 16, 32, 64]
    },
    "anchor": {
        "num_anchors": 9,
        "strides": [4, 8, 16, 32, 64],
        "scales": [[32], [64], [128], [256], [512]],
        "aspect_ratios": [2.0, 1.0, 0.5],
    },
    "assigner": {
        "name": "max_iou_assigner",
        "pos_iou_thresh": 0.7,
        "neg_iou_thresh": 0.3,
        "bbox_encoder": None
    },
    "sampler": None,
    "loss": {
        "label_loss": {
            "name": "cross_entropy",
            "use_sigmoid": True,
            "weight": 1.,
            "reduction": "none"
        },
        "bbox_loss": {
            "name": "smooth_l1",
            "delta": 1.,
            "weight": 1.,
            "reduction": "none"
        },
        "weight_decay": 0.0001
    },
    "train": {
        "batch_size": 2,
        "input_size": [640, 640],
        "dataset": "wider_face",
        "samples": 12876,
        "num_classes": 1,
        "augmentation": "hybrid",  # The data augmentation type, e.g. ssd, data_anchor_sampling,
        # hybrid(ssd and data_anchor_sampling) or auto_augment(auto augmentation).
        "normalize_gt_bbox": True,
        "dataset_dir": "/home/bail/Data/data1/Dataset/WiderFace/WIDER_train",
        "pretrained_weights_path": "",
        "optimizer": {
            "name": "momentum",
            "momentum": 0.9,
        },
        "total_epochs": 25,
        "learning_rate": {
            "name": "step",
            "warmup_learning_rate": 0.008,
            "warmup_epochs": 5,
            "init_learning_rate": 0.08,
            "values": [0.008, 0.0008],
            "boundaries": [15, 20],
        },
        "checkpoint": {
            "path": "",
            "prefix": "",
        },
        "frozen_variable_stages": [],
        "gradient_clip_norm": 0.0,
    },
    "eval": {
        "batch_size": 1,
        "samples": 5000,
        "dataset_dir": "",
    },
    "postprocess": {
        "max_total_size": 1000,
        "nms_iou_threshold": 0.5,
        "score_threshold": 0.5,
    }
}


RESTRICTIONS = [
    "head.num_anchors == anchor.num_anchors",
    "head.anchor_strides == anchor.strides",
    "head.num_classes == train.num_classes",
    "backbone.normalization.axis == head.normalization.axis",
    "backbone.normalization.axis == neck.normalization.axis"
]
