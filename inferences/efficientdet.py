import cv2
import time
import random
import numpy as np
import tensorflow as tf
from core.bbox import Delta2Box
from configs import build_configs
from core.layers import build_nms
from detectors import build_detector


def _generate_anchor_configs(min_level, max_level, num_scales, aspect_ratios):
    """Generates mapping from output level to a list of anchor configurations.
        A configuration is a tuple of (num_anchors, scale, aspect_ratio).
        Args:
            min_level: integer number of minimum level of the output feature pyramid.
            max_level: integer number of maximum level of the output feature pyramid.
            num_scales: integer number representing intermediate scales added
                on each level. For instances, num_scales=2 adds two additional
                anchor scales [2^0, 2^0.5] on each level.
            aspect_ratios: list of tuples representing the aspect ratio anchors added
                on each level. For instances, aspect_ratios =
                [(1, 1), (1.4, 0.7), (0.7, 1.4)] adds three anchors on each level.
        Returns:
            anchor_configs: a dictionary with keys as the levels of anchors and
            values as a list of anchor configuration.
    """
    anchor_configs = {}
    for level in range(min_level, max_level + 1):
        anchor_configs[level] = []
        for scale_octave in range(num_scales):
            for aspect in aspect_ratios:
                anchor_configs[level].append((2**level, scale_octave / float(num_scales), aspect))
    return anchor_configs


def _generate_anchor_boxes(image_size, anchor_scale, anchor_configs):
    """Generates multiscale anchor boxes.
        Args:
            image_size: integer number of input image size. The input image has the
            same dimension for width and height. The image_size should be divided by
            the largest feature stride 2^max_level.
            anchor_scale: float number representing the scale of size of the base
            anchor to the feature stride 2^level.
            anchor_configs: a dictionary with keys as the levels of anchors and
            values as a list of anchor configuration.
        Returns:
            anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all
            feature levels.
        Raises:
            ValueError: input size must be the multiple of largest feature stride.
    """
    boxes_all = []
    for _, configs in anchor_configs.items():
        boxes_level = []
        for config in configs:
            stride, octave_scale, aspect = config
            if image_size % stride != 0:
                raise ValueError('input size must be divided by the stride.')
            base_anchor_size = anchor_scale * stride * 2**octave_scale
            anchor_size_x_2 = base_anchor_size * aspect[0] / 2.0
            anchor_size_y_2 = base_anchor_size * aspect[1] / 2.0

            x = np.arange(stride / 2, image_size, stride)
            y = np.arange(stride / 2, image_size, stride)
            xv, yv = np.meshgrid(x, y)
            xv = xv.reshape(-1)
            yv = yv.reshape(-1)

            boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                               yv + anchor_size_y_2, xv + anchor_size_x_2))
            boxes = np.swapaxes(boxes, 0, 1)
            boxes_level.append(np.expand_dims(boxes, axis=1))
        # concat anchors on the same level to the reshape NxAx4
        boxes_level = np.concatenate(boxes_level, axis=1)
        boxes_all.append(boxes_level.reshape([-1, 4]))

    anchor_boxes = np.vstack(boxes_all)
    return anchor_boxes


class EfficientDet(tf.keras.Model):
    def __init__(self, **kwargs):
        super(EfficientDet, self).__init__(**kwargs)
        cfg = build_configs("efficientdet")

        self.input_size = cfg.val.dataset.input_size[0]
        self.model = build_detector(cfg.detector, cfg=cfg).model
        self.nms = build_nms("fast_non_max_suppression", cfg)
        self.delta2box = Delta2Box(mean=None, std=None)
        anchors_configs = _generate_anchor_configs(3, 7, 3, [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        anchors = _generate_anchor_boxes(cfg.val.dataset.input_size[0], 4, anchors_configs)
        self.anchors = tf.convert_to_tensor(anchors, tf.float32)

    @tf.function
    def call(self, inputs):
        predicted_boxes, predicted_labels = self.model(inputs, training=False)
        predicted_boxes = self.delta2box(self.anchors, predicted_boxes)
        predicted_boxes *= tf.clip_by_value(1. / self.input_size, 0, 1)
        predicted_scores = tf.nn.sigmoid(predicted_labels)
        # tf.print(predicted_boxes)
        # tf.print(tf.reduce_max(predicted_scores))

        return self.nms(predicted_boxes, predicted_scores)


coco_id_mapping = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
    73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush',
}


def save_model():
    efficientdet = EfficientDet()
    efficientdet(tf.random.uniform([1] + [efficientdet.input_size] * 2 + [3], 0, 255), training=False)
    tf.saved_model.save(efficientdet, "./saved_model/efficientdet/1/")


def random_color(seed=None):
    random.seed(seed % 32)
    levels = range(32, 256, 32)

    return tuple(random.choice(levels) for _ in range(3))


def draw(img, box, label, score, names_dict, color=None):
    c1 = (int(box[1]), int(box[0]))
    c2 = (int(box[3]), int(box[2]))
    
    if int(label) not in names_dict:
        return img
    label = names_dict[int(label)]
    text = label + ":{:.2f}".format(float(score))
    # score = names_dict[detection[4].float()]

    # color = random.choice(self._colors)
    img = cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] - t_size[1] - 4
    img = cv2.rectangle(img, c1, c2, color, -1)
    # cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    img = cv2.putText(img, text, (c1[0], c1[1]), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0], 1)

    return img


def inference():
    loaded = tf.saved_model.load("./saved_model/efficientdet/1", tags=[tf.saved_model.SERVING])

    infer = loaded.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    infer = loaded.signatures["serving_default"]
    # loaded = tf.saved_model.load("./efficientdet/1/")

    video = cv2.VideoCapture("/home/bail/Workspace/LuZhengTongV4/data/18_parking20200206090159.mp4")
    
    coco_inds = list(coco_id_mapping.keys())
    while True:
        return_value, frame = video.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("No image!")
        frame_size = frame.shape[:2]
        input_size = 512
        image_data = cv2.resize(np.copy(frame), (input_size, input_size))
        image_data = tf.convert_to_tensor(image_data[np.newaxis, ...].astype(np.float32))
        prev_time = time.time()
        # print(infer(image_data))
        outputs = infer(image_data)
        # num = outputs["output_4"].numpy()[0]
        # boxes = outputs["output_1"].numpy()[0][:num]
        # scores = outputs["output_2"].numpy()[0][:num]
        # classes = outputs["output_3"].numpy()[0][:num]
        num = outputs["valid_detections"].numpy()[0]
        boxes = outputs["nmsed_boxes"].numpy()[0][:num]
        scores = outputs["nmsed_scores"].numpy()[0][:num]
        classes = outputs["nmsed_classes"].numpy()[0][:num]
        for i in range(num):
            box = boxes[i] * np.array([frame_size[0], frame_size[1], frame_size[0], frame_size[1]])
            cls = classes[i]
            frame = draw(frame, box, cls + 1, scores[i], coco_id_mapping, random_color(int(cls)))

        curr_time = time.time()
        exec_time = curr_time - prev_time

        info = "time: %.2f ms" % (1000 * exec_time)
        cv2.putText(frame, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        result = cv2.resize(result, (1024, 576))
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    save_model()

