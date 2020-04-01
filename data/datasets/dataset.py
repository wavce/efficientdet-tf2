import math
import tensorflow as tf
from data.augmentations import Compose 
from core.anchors import AnchorGenerator
from core.assigners import build_assigner

class Dataset(object):
    def __init__(self, 
                 dataset_dir, 
                 training=True,
                 batch_size=32, 
                 input_size=(512, 512), 
                 augmentation=[],
                 **kwargs):
        self.dataset_dir = dataset_dir
        self.training = training
        self.batch_size = batch_size
        self.input_size = input_size

        self.num_images_per_record = 10000

        self.tf_record_sources = None

        if "anchor" in kwargs and kwargs["anchor"]:
            self.anchor_kwargs = kwargs["anchor"]
            self.min_level = self.anchor_kwargs["min_level"]
            self.max_level = self.anchor_kwargs["max_level"]
            self.anchor_generator = AnchorGenerator()

        self.assigner_kwargs = kwargs["assigner"]
        self.min_level = self.assigner_kwargs.pop("min_level")
        self.max_level = self.assigner_kwargs.pop("max_level")
       
        self._use_fcos_assigner = self.assigner_kwargs["assigner"] == "fcos_assigner"
        self._use_mask_assigner = "mask" in self.assigner_kwargs["assigner"]
        self.assigner = build_assigner(**self.assigner_kwargs)

        self.augment = Compose(augmentation) if augmentation is not None else None

    def is_valid_jpg(self, jpg_file):
        with open(jpg_file, 'rb') as f:
            f.seek(-2, 2)
            buf = f.read()
            f.close()
            return buf == b'\xff\xd9'  # 判定jpg是否包含结束字段
    @property
    def rgb_mean(self):
        return tf.constant([0.485 * 255, 0.456 * 255, 0.406 * 255], dtype=tf.float32)
    
    @property
    def rgb_std(self):
        return tf.constant([0.229 * 255, 0.224 * 255, 0.225 * 255], dtype=tf.float32)

    def create_tfrecord(self):
        raise NotImplementedError()

    def parser(self, serialized):
        raise NotImplementedError()

    def _build_targets_with_mask(self, gt_boxes, gt_labels):
        target_boxes = []
        target_labels = []
        target_attentions = []
        total_anchors = dict()
        for i, level in enumerate(range(self.min_level, self.max_level + 1)):
            strides = self.anchor_kwargs["strides"][i]
            feat_h = int(math.ceil(self.input_size[0] // strides))
            feat_w = int(math.ceil(self.input_size[1] // strides))
            anchors = self.anchor_generator(feature_map_size=[feat_h, feat_w], 
                                            scales=self.anchor_kwargs["scales"][i], 
                                            aspect_ratios=self.anchor_kwargs["aspect_ratios"][i],  
                                            strides=strides)

            t_boxes, t_labels, t_attentions = self.assigner(gt_boxes, gt_labels, anchors)

            target_boxes.append(t_boxes)
            target_labels.append(t_labels)
            target_attentions["level_%d" % level] = t_attentions
            total_anchors["level_%d" % level] = anchors
        
        target_boxes = tf.concat(target_boxes, 0)
        target_labels = tf.concat(target_labels, 0)
        total_anchors = tf.concat(total_anchors, 0)

        return dict(target_boxes=target_boxes,
                    target_labels=target_labels,
                    target_attentions=target_attentions,
                    total_anchors=total_anchors)

    def _build_targets(self, gt_boxes, gt_labels):
        total_anchors = []
        for i, level in enumerate(range(self.min_level, self.max_level + 1)):
            strides = self.anchor_kwargs["strides"][i]
            feat_h = int(math.ceil(self.input_size[0] // strides))
            feat_w = int(math.ceil(self.input_size[1] // strides))
            anchors = self.anchor_generator(feature_map_size=[feat_h, feat_w], 
                                            scales=self.anchor_kwargs["scales"][i], 
                                            aspect_ratios=self.anchor_kwargs["aspect_ratios"][i], 
                                            strides=strides)
            total_anchors.append(anchors)
        
        total_anchors = tf.concat(total_anchors, 0)
        target_boxes, target_labels = self.assigner(gt_boxes, gt_labels, total_anchors)
        
        return dict(target_boxes=target_boxes,
                    target_labels=target_labels,
                    total_anchors=total_anchors)

    def _build_fcos_targets(self, gt_boxes, gt_labels):
        target_boxes = []
        target_labels = []
        target_centerness = []

        grid_x = []
        grid_y = []
        for i, level in enumerate(range(self.min_level, self.max_level + 1)):
            strides = 2 ** level
            feat_h = int(math.ceil(self.input_size[0] // strides))
            feat_w = int(math.ceil(self.input_size[1] // strides))

            xx, yy = tf.meshgrid(tf.range(feat_w), tf.range(feat_h))
            xx = tf.cast(xx, tf.float32) * strides + 0.5 * strides
            yy = tf.cast(yy, tf.float32) * strides + 0.5 * strides
  
            t_boxes, t_labels, t_centerness = self.assigner(
                gt_boxes=gt_boxes, gt_labels=gt_labels, 
                grid_y=yy, grid_x=xx, strides=strides,
                object_size_of_interest=self.assigner_kwargs["object_sizes_of_interest"][i])

            yy = tf.reshape(yy, [feat_h * feat_w])
            xx = tf.reshape(xx, [feat_h * feat_w])
            yy /= (feat_h * strides)
            xx /= (feat_w * strides)
            target_boxes.append(t_boxes)
            target_labels.append(t_labels)
            target_centerness.append(t_centerness)
            grid_x.append(xx)
            grid_y.append(yy)
        
        target_boxes = tf.concat(target_boxes, 0)
        target_labels = tf.concat(target_labels, 0)
        target_centerness = tf.concat(target_centerness, 0)
        grid_y = tf.concat(grid_y, 0)
        grid_x = tf.concat(grid_x, 0)

        return dict(target_boxes=target_boxes,
                    target_labels=target_labels,
                    target_centerness=target_centerness,
                    grid_y=grid_y,
                    grid_x=grid_x)

    def dataset(self):
        with tf.device("/cpu:0"):
            dataset = tf.data.TFRecordDataset(self.tf_record_sources)
            dataset = dataset.map(map_func=self.parser)
            if self.training:
                dataset = dataset.repeat() 
                dataset = dataset.shuffle(buffer_size=self.batch_size * 40)
            dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True)
           
            return dataset.prefetch(tf.data.experimental.AUTOTUNE)