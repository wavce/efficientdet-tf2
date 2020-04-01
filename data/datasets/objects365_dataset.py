import os
import cv2
import json
import tqdm
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from data.datasets import Dataset


class Objects365Dataset(Dataset):
    def __init__(self, 
                 dataset_dir, 
                 training=True,
                 batch_size=32, 
                 input_size=(512, 512), 
                 augmentation=dict(),
                 **kwargs):
        super(Objects365Dataset, self).__init__(dataset_dir, training, batch_size, input_size, augmentation, **kwargs)

        self.tf_record_sources = glob.glob(os.path.join(self.dataset_dir, "*.tfrecord"))

    def is_valid_jpg(self, jpg_file):
        with open(jpg_file, 'rb') as f:
            f.seek(-2, 2)
            buf = f.read()
            f.close()
            return buf == b'\xff\xd9'  # 判定jpg是否包含结束字段

    def create_tfrecord(self):
        if self.training:
            tfrecord_writer = tf.io.TFRecordWriter(
                path=os.path.join(self.dataset_dir, "train/" + "0.tfrecord"))
        else:
            tfrecord_writer = tf.io.TFRecordWriter(
                path=os.path.join(self.dataset_dir, "val/" + ".tfrecord"))

        n = 0
        json_file = os.path.join(self.dataset_dir, "training" + ".json")
        with open(json_file, "r") as f:
            data = json.load(f)
            annotations = pd.DataFrame(data["annotations"])
            images_info = data["images"]

        del data
        f.close()
        for img_dict in tqdm.tqdm(images_info):
            img_id = img_dict["id"]
            height = img_dict["height"]
            width = img_dict["width"]
            img_name = img_dict["file_name"]

            if "jpg" not in img_name:
                continue

            img_name = os.path.join(self.dataset_dir, "images", "training", img_name)
            if not os.path.exists(img_name):
                continue

            if not self.is_valid_jpg(img_name):
                continue

            if n % self.num_images_per_record == 0:
                tfrecord_writer.close()
                tfrecord_writer = tf.io.TFRecordWriter(path=os.path.join(
                    self.dataset_dir, self.phase + "{}.tfrecord".format(n // self.num_images_per_record + 1)))

            anns = annotations[annotations["image_id"] == img_id]

            category_id = anns["category_id"].tolist()
            is_crowd = anns["iscrowd"].tolist()
            count = len(anns)
            if count == 0:
                continue
            bbox = anns["bbox"].tolist()
            if len(bbox) == 0:
                continue
            bbox = np.array(bbox)
            x, y, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]

            with tf.io.gfile.GFile(img_name, 'rb') as gf:
                encoded_image = gf.read()

            n += 1
            example = tf.train.Example(features=tf.train.Features(feature={
                "image/encoded": tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image])),
                "image/height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                "image/width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                "image/object/categories": tf.train.Feature(int64_list=tf.train.Int64List(value=category_id)),
                "image/object/is_crowd": tf.train.Feature(int64_list=tf.train.Int64List(value=is_crowd)),
                "image/object/count": tf.train.Feature(int64_list=tf.train.Int64List(value=[count])),
                "image/object/bbox/x": tf.train.Feature(float_list=tf.train.FloatList(value=x.tolist())),
                "image/object/bbox/y": tf.train.Feature(float_list=tf.train.FloatList(value=y.tolist())),
                "image/object/bbox/width": tf.train.Feature(float_list=tf.train.FloatList(value=w.tolist())),
                "image/object/bbox/height": tf.train.Feature(float_list=tf.train.FloatList(value=h.tolist()))}))
            tfrecord_writer.write(example.SerializeToString())

        tfrecord_writer.close()
        print("Total write %d images to tfrecord." % n)

    def parser(self, serialized):
        key_to_features = {
            'image/width': tf.io.FixedLenFeature([], tf.int64, 0),
            'image/height': tf.io.FixedLenFeature([], tf.int64, 0),
            'image/encoded': tf.io.FixedLenFeature([], tf.string, ''),
            'image/object/count': tf.io.FixedLenFeature([], tf.int64, 0),
            'image/object/categories': tf.io.VarLenFeature(tf.int64),
            'image/object/bbox/x': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/y': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/width': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/height': tf.io.VarLenFeature(tf.float32)
        }

        parsed_features = tf.io.parse_single_example(
            serialized, features=key_to_features, name='parse_features')

        image = tf.image.decode_image(parsed_features['image/encoded'])

        image = tf.cond(tf.equal(tf.shape(image)[-1], 1),
                        true_fn=lambda: tf.tile(image, [1, 1, 3]),
                        false_fn=lambda: image)
        img_width = tf.cast(parsed_features['image/width'], tf.int32)
        img_height = tf.cast(parsed_features['image/height'], tf.int32)
        image = tf.reshape(image, [img_height, img_width, 3])
        image = tf.cast(image, tf.float32)

        labels = tf.cast(tf.sparse.to_dense(parsed_features['image/object/categories']), tf.float32)
        # labels += 1.
        x = tf.cast(tf.sparse.to_dense(parsed_features['image/object/bbox/x']), tf.float32)
        y = tf.cast(tf.sparse.to_dense(parsed_features['image/object/bbox/y']), tf.float32)
        w = tf.cast(tf.sparse.to_dense(parsed_features['image/object/bbox/width']), tf.float32)
        h = tf.cast(tf.sparse.to_dense(parsed_features['image/object/bbox/height']), tf.float32)
    
        # float_img_h = tf.cast(img_height, tf.float32)
        # float_img_w = tf.cast(img_width, tf.float32)

        # x *= float_img_w
        # y *= float_img_h
        # w *= float_img_w
        # h *= float_img_h

        boxes = tf.stack([y, x, y + h, x + w], axis=-1)
        boxes = tf.clip_by_value(boxes, 0, 1)

        if self.training:
            image, boxes, labels = self.augment(image, boxes, labels)
        else:
            image = tf.image.resize(image, self.input_size)

        if self._use_fcos_assigner:
            image_info = self._build_fcos_targets(boxes, labels)
        elif self._use_mask_assigner:
            image_info = self._build_targets_with_mask(boxes, labels)
        else:
            image_info = self._build_targets(boxes, labels)
        
        num_labels = tf.size(labels)
        if num_labels < 200:
            labels = tf.concat([labels, tf.zeros([200 - num_labels], labels.dtype)], 0)
            boxes = tf.concat([boxes, tf.zeros([200 - num_labels, 4], boxes.dtype)], 0)
        else:
            labels = labels[:200]
            boxes = boxes[:200]

        image_info["gt_boxes"] = boxes
        image_info["gt_labels"] = labels

        return image, image_info


def main():
    count = 0
    input_size = 800
    dataset_dir = "/home/bail/Data/data1/Dataset/Objects365/train"
    
    assigner = {
        "assigner": "fcos_assigner",
        "object_sizes_of_interest": [[0, 64],
                                     [64, 128],
                                     [128, 256],
                                     [256, 512],
                                     [512, 1e10]],
        "sampling_radius": 0.,
        "min_level": 3,
        "max_level": 7,
    }
    augmentation = [
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
    dataset = Objects365Dataset(dataset_dir, True, 1, (input_size, input_size), augmentation=augmentation, assigner=assigner).dataset()
    for images, image_info in dataset.take(100):
        count += 1
        labels = image_info["gt_labels"]
        boxes = image_info["gt_boxes"]
        print(count, labels)
        # plt.figure()
        for j in range(1):
            count += 1
            img = images[j]
            h, w, _ = img.shape
            img = img.numpy()
            img = img.astype(np.uint8)

            box = boxes[j].numpy()
            # print('\n', count)
            for l in box:
                pt1 = (int(l[1]), int(l[0]))
                pt2 = (int(l[3]), int(l[2]))
                # print(pt1, pt2)

                img = cv2.rectangle(img, pt1=pt1, pt2=pt2, thickness=1, color=(0, 255, 0))

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("image", img)
            cv2.waitKey(0)
        #    img = cv2.cvtColor(img, code=cv2.COLOR_RGB2BGR)
        #    cv2.imwrite('./label%d.jpg' % count, img)


def progress(percent, width=50):
    """进度打印功能"""
    if percent >= 100:
        percent = 100
        show_str = ('[%%-%ds]' % width) % (int(width * percent / 100) * "#")  # 字符串拼接的嵌套使用
        print('\r%s %d%% ' % (show_str, percent), end='')


if __name__ == '__main__':
    # create_tfrecord(FLAGS.dataset_dir, "train")
    main()
