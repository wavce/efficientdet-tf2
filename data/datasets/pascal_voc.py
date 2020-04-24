import os
import glob
import random
import tensorflow as tf
from xml.etree import ElementTree as ET
from data.datasets.dataset import Dataset


class PASCALVOC(Dataset):
    def __init__(self,
                 data_dir,
                 input_size,
                 batch_size,
                 training=True,
                 augmentation="auto_augment",
                 **kwargs):
        super(PASCALVOC, self).__init__(data_dir=data_dir, 
                                        training=training,
                                        input_size=input_size,
                                        batch_size=batch_size,
                                        augmentation=augmentation,
                                        **kwargs)
        assert os.path.exists(data_dir), "%s not exits." % data_dir

    def create_tfrecord(self, data_dir, num_shards):
        image_files = glob.glob(os.path.join(data_dir, "*/*.jpg"))
        image_files = [f for f in image_files if os.path.exists(f) and os.path.exists(f.replace("jpg", "xml"))]
        random.shuffle(image_files)

        num_images = len(image_files)
        max_length_per_tfrecord = 10000 if num_images > 10000 else num_images
        num_tfrecord = (num_images // max_length_per_tfrecord 
                        if num_images % max_length_per_tfrecord == 0 
                        else num_images // max_length_per_tfrecord + 1)

        if self.training:
            tfrecord_dir = os.path.join(self.data_dir, "train")
        else:
            tfrecord_dir = os.path.join(self.data_dir, "eval")
        
        if not os.path.exists(tfrecord_dir):
                os.mkdirs(tfrecord_dir)
        tfrecord_path = [os.path.join(tfrecord_dir,  "luzheng.tfrecord-%d-of-%d" % (i + 1, num_tfrecord))
                         for i in range(num_tfrecord)]

        if len(image_files) == 0:
            raise ValueError("images not exists.")
        
        count = 0
        tfrecord_count = 0
        writer = tf.io.TFRecordWriter(tfrecord_path[tfrecord_count])
        for i, img_path in enumerate(image_files):
            if not os.path.exists(img_path):
                print(count, img_path, "not exists.")
                continue
            
            xml_path = img_path.replace("jpg", "xml")
            if not os.path.exists(xml_path):
                print(count, xml_path, "not exists.")
                continue

            num_obj = 0
            x1s = []
            y1s = []
            x2s = []
            y2s = []
            labels = []
            
            tree = ET.ElementTree(file=xml_path)
            root = tree.getroot()
            height = int(root.find("size").find("height").text)
            width = int(root.find("size").find("width").text)
            channels = int(root.find("size").find("depth").text)

            objects = root.findall("object")
            for obj in objects:
                name = obj.find("name").text
                if "trunk" in name:
                    name = "truck"
                name = name.strip()
                if name == "w" or name == "D":
                    continue

                labels.append(NAMES.index(name) + 1)
                
                x1s.append(float(obj.find("bndbox").find("xmin").text))
                y1s.append(float(obj.find("bndbox").find("ymin").text))
                x2s.append(float(obj.find("bndbox").find("xmax").text))
                y2s.append(float(obj.find("bndbox").find("ymax").text))

            if all((x1s[i] == 0 and y1s[i] == 0 and x2s[i] == 0 and y2s[i] == 0)for i in range(len(x1s))):
                print(img_path, "have no box.")
                continue

            with tf.io.gfile.GFile(img_path, "rb") as gf:
                encoded_image = gf.read()

            example = tf.train.Example(features=tf.train.Features(
                feature={
                    "image/encoded": tf.train.Feature(bytes_list=self._bytes_list([encoded_image])),
                    "image/height": tf.train.Feature(int64_list=self._int64_list([height])),
                    "image/width": tf.train.Feature(int64_list=self._int64_list([width])),
                    "image/channels": tf.train.Feature(int64_list=self._int64_list([channels])),
                    "image/object/bbox/x1": tf.train.Feature(float_list=self._float_list(x1s)),
                    "image/object/bbox/y1": tf.train.Feature(float_list=self._float_list(y1s)),
                    "image/object/bbox/x2": tf.train.Feature(float_list=self._float_list(x2s)),
                    "image/object/bbox/y2": tf.train.Feature(float_list=self._float_list(y2s)),
                    "image/object/bbox/label": tf.train.Feature(int64_list=self._int64_list(labels))
                }))

            writer.write(example.SerializeToString())
            count += 1
            
            if count % max_length_per_tfrecord == 0:
                print("Wrote %d images to %s." % (max_length_per_tfrecord, tfrecord_path[tfrecord_count]))
                tfrecord_count += 1
                writer.close()
                writer = tf.io.TFRecordWriter(tfrecord_path[tfrecord_count])

        writer.close()
        print("Wrote %d images to %s." % (num_images - max_length_per_tfrecord * tfrecord_count, 
                                          tfrecord_path[tfrecord_count]))
        print("Total write %d images to %s." % (count, tfrecord_path))

    def parse(self, serialized):
        key_to_features = {
            "image/encoded": tf.io.FixedLenFeature([], tf.string, ""),
            "image/height": tf.io.FixedLenFeature([], tf.int64, 0),
            "image/width": tf.io.FixedLenFeature([], tf.int64, 0),
            "image/channels": tf.io.FixedLenFeature([], tf.int64, 0),
            "image/object/bbox/x1": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/y1": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/x2": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/y2": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/label": tf.io.VarLenFeature(tf.int64)
        }
        parsed_features = tf.io.parse_single_example(serialized, features=key_to_features)

        img_height = parsed_features["image/height"]
        img_width = parsed_features["image/width"]
        channels = parsed_features["image/channels"]

        image = tf.io.decode_image(parsed_features["image/encoded"])
        image = tf.reshape(image, [img_height, img_width, channels])
        image = tf.cast(image, tf.float32)

        x1 = tf.cast(tf.sparse.to_dense(parsed_features["image/object/bbox/x1"]), tf.float32)
        y1 = tf.cast(tf.sparse.to_dense(parsed_features["image/object/bbox/y1"]), tf.float32)
        x2 = tf.cast(tf.sparse.to_dense(parsed_features["image/object/bbox/x2"]), tf.float32)
        y2 = tf.cast(tf.sparse.to_dense(parsed_features["image/object/bbox/y2"]), tf.float32)
        labels = tf.cast(tf.sparse.to_dense(parsed_features["image/object/bbox/label"]), tf.int64)

        bbox = tf.stack([y1, x1, y2, x2], axis=-1)
        normalizer = tf.convert_to_tensor([self.input_size[0],
                                           self.input_size[1],
                                           self.input_size[0],
                                           self.input_size[1]], tf.float32)

        if self.augment:
            image, bbox, labels = self.augment(image, bbox, labels)
        else:
            image = tf.image.resize(image, self.input_size)
            bbox /= tf.convert_to_tensor([img_height, img_width, img_height, img_width], tf.float32)
            bbox *= normalizer

        if self.normalize_box:
            bbox *= (1. / normalizer)
            bbox = tf.clip_by_value(bbox, 0, 1)

        return image, bbox, labels

    # @property
    def dataset(self):
        tfrecord_path = glob.glob(os.path.join(self.data_dir, "*.tfrecord*"))
        if len(tfrecord_path) == 0:
            raise ValueError("Cannot find tfrecord in %s" % self.data_dir)

        with tf.device("/cpu:0"):
            ds = tf.data.TFRecordDataset(tfrecord_path)
            ds = ds.map(map_func=self.parse)

            if self.training:
                ds = ds.shuffle(self.batch_size * 10)

            ds = ds.padded_batch(batch_size=self.batch_size,
                                 padded_shapes=([None, None, 3], [None, 4], [None]),
                                 drop_remainder=False)

            return ds.prefetch(tf.data.experimental.AUTOTUNE)


def main():
    count = 0
    input_size = (640, 640)
    num_scales = 3
    base_scale = 4
    anchor_strides = [8, 16, 32, 64, 128]
    anchor_scales = [[2 ** (i / num_scales) * s * base_scale
                      for i in range(num_scales)] for s in anchor_strides]
    wider_face = LuZhengDataSet(data_dir="/home/deepblue/Data/yaoyq/train_data/lz1216/train",
                                input_size=input_size,
                                batch_size=1,
                                training=True,
                                normalize_box=False,
                                augmentation="hybrid",
                                anchor_scales=anchor_scales)
    # wider_face.create_tfrecord()

    min_scale = 9999
    max_scale = 0
    dataset = wider_face.dataset
    for images, boxes, labels in dataset().take(100):
        count += 1
        print('\n', count, boxes.shape)
        # plt.figure()
        images = images.numpy()
        boxes = boxes.numpy()
        # print(boxes[np.logical_not(np.all(boxes == 0, -1))])
        box_hw = boxes[..., 2:4] - boxes[..., 0:2]
        # if np.min(box_hw) < min_scale:
        #     min_scale = np.min(box_hw)
        # if np.max(box_hw) > max_scale:
        #     max_scale = np.max(box_hw)
        # print(min_scale, max_scale)

        for j in range(1):
            h, w, _ = images[j].shape
            img = images[j]
            img = img.astype(np.uint8)
            box = boxes[j]
            for l in box:
                pt1 = (int(l[1]), int(l[0]))
                pt2 = (int(l[3]), int(l[2]))
        
                img = cv2.rectangle(img, pt1=pt1, pt2=pt2, thickness=1, color=(0, 255, 0))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("image", img)
            cv2.waitKey(0)
            img = cv2.cvtColor(img, code=cv2.COLOR_RGB2BGR)
            cv2.imwrite('./label%d.jpg' % count, img)


if __name__ == '__main__':
    main()


