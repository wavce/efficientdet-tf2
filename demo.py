import os
import cv2
import time
import random
import argparse
import numpy as np
import tensorflow as tf


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


def preprocess(image, input_scale=None):
    height, width, _ = image.shape
    ratio = height / width
    if height < width:
        height = input_scale if input_scale else int((height // 128) * 128)
        width = int(height / ratio // 128 * 128)
    else:
        width = input_scale if input_scale else int(width // 128 * 128)
        height = int(ratio * width // 128 * 128)
    
    image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (width, height))

    return image


def inference(saved_model_dir, video_path, input_size=512):
    loaded = tf.saved_model.load(saved_model_dir, tags=[tf.saved_model.SERVING])

    infer = loaded.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    infer = loaded.signatures["serving_default"]

    img = cv2.imread("./data/images/img.png")
    height, width = img.shape[0:2]
    img_data = tf.image.convert_image_dtype(preprocess(img.copy(), input_size)[np.newaxis, ...], tf.float32)
    outputs = infer(img_data)
    num = outputs["valid_detections"].numpy()[0]
    boxes = outputs["nmsed_boxes"].numpy()[0][:num]
    scores = outputs["nmsed_scores"].numpy()[0][:num]
    classes = outputs["nmsed_classes"].numpy()[0][:num]
    for i in range(num):
        box = boxes[i] * np.array([height, width, height, width])
        cls = classes[i]
        img = draw(img, box, cls + 1, scores[i], coco_id_mapping, random_color(int(cls)))
    
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if not os.path.exists(video_path):
        return

    video = cv2.VideoCapture(video_path)
    
    while True:
        return_value, frame = video.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("No image!")
        frame_size = frame.shape[:2]
        image_data = tf.image.convert_image_dtype(preprocess(frame.copy(), input_size)[np.newaxis, ...], tf.float32)
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


def main():
    parser = argparse.ArgumentParser(description="Demo args")
    parser.add_argument("--saved_model_dir", default="./saved_model/efficientdet/1", type=str)
    parser.add_argument("--video_path", default="../2.mp4", type=str)
    parser.add_argument("--input_size", default=None, type=int)

    args = parser.parse_args()

    saved_model_dir = args.saved_model_dir
    video_path = args.video_path

    inference(saved_model_dir, video_path, args.input_size)

if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device=device, enable=True)

    main()
