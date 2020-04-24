import os
import io
import json
import hashlib
import logging
import argparse
import PIL.Image
import pycocotools
import collections
import numpy as np
import multiprocessing
import tensorflow as tf
from pycocotools import mask
from data.datasets import Dataset


class COCODataset(Dataset):
    def __init__(self, dataset_dir, training=True, batch_size=32, input_size=(512,512), augmentation=[], **kwargs):
        super().__init__(dataset_dir, training=training, batch_size=batch_size, input_size=input_size, augmentation=augmentation, **kwargs)

    def create_tf_example(self, 
                          image, 
                          image_dir, 
                          bbox_annotations=None,
                          category_index=None,
                          caption_annotations=None,
                          include_masks=False):
        """Converts image and annotations to a tf.Example proto.

        Args:
            image: dict with keys: [u'license', u'file_name', u'coco_url', u'height',
                u'width', u'date_captured', u'flickr_url', u'id']
            image_dir: directory containing the image files.
            bbox_annotations: list of dicts with keys: [u'segmentation', u'area', u'iscrowd',
                u'image_id', u'bbox', u'category_id', u'id'] Notice that bounding box
                coordinates in the official COCO dataset are given as [x, y, width,
                height] tuples using absolute coordinates where x, y represent the
                top-left (0-indexed) corner.  This function converts to the format
                expected by the Tensorflow Object Detection API (which is which is
                [ymin, xmin, ymax, xmax] with coordinates normalized relative to image
                size).
            category_index: a dict containing COCO category information keyed by the
                'id' field of each category.  See the label_map_util.create_category_index
                function.
            caption_annotations: list of dict with keys: [u'id', u'image_id', u'str'].
            include_masks: Whether to include instance segmentations masks
                (PNG encoded) in the result. default: False.

        Returns:
            example: The converted tf.Example
            num_annotations_skipped: Number of (invalid) annotations that were ignored.

        Raises:
            ValueError: if the image pointed to by data['filename'] is not a valid JPEG
        """
        img_height = image["height"]
        img_width = image["width"]
        filename = image["filename"]
        img_id = image["id"]

        img_path = os.path.join(image_dir, filename)
        with tf.io.gfile.GFile(image_dir, img_path) as fid:
            encoded_jpg = fid.read()
        
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        key = hashlib.sha256(encoded_jpg_io).hexdigest()
        feature_dict = {
            "image/height": self._int64_list(img_height),
            "image/width": self._int64_list(img_width),
            "image/filename": self._bytes_list(filename.encode("utf8")),
            "image/source_id": self._bytes_list(str(img_id).encode("utf8")),
            "image/key/sha256": self._bytes_list(key.encode("utf8")),
            "image/encoded": self._bytes_list(encoded_jpg),
            "image/format": self._bytes_list("jpeg".encode("utf8"))
        }

        num_annotations_skipped = 0
        if bbox_annotations:
            xmin = []
            ymin = []
            xmax = []
            ymax = []
            is_crowd = []
            category_names = []
            category_ids = []
            area = []
            encoded_mask_png = []
            for object_annotations in bbox_annotations:
                (x, y, w, h) = tuple(object_annotations["bbox"])

                if w <= 0 or h <= 0:
                    num_annotations_skipped += 1
                    continue
                if x + w > img_width or y + h > img_height:
                    num_annotations_skipped += 1
                    continue
                xmin.append(float(x) / img_width)
                xmax.append(float(x + w) / img_width)
                ymin.append(float(y) / img_height)
                ymax.append(float(y + h) / img_height)
                is_crowd.append(object_annotations["iscrowd"])
                category_id = int(object_annotations["category_id"])
                category_ids.append(category_id)
                category_names.append(category_index[category_id]["name"].encode("utf8"))
                area.append(object_annotations["area"])

                if include_masks:
                    run_len_encoding = mask.frPyObjects(
                        object_annotations["segmentation"], img_height, img_width)
                    binary_mask = mask.decode(run_len_encoding)

                    if not object_annotations["iscrowd"]:
                        binary_mask = np.amax(binary_mask, axis=2)
                    pil_image = PIL.Image.fromarray(binary_mask)
                    output_io = io.BytesIO()
                    pil_image.save(output_io, format="PNG")
                    encoded_mask_png.append(output_io.getvalue())
            feature_dict.update({
                "image/object/bbox/xmin": self._float_list(xmin),
                "image/object/bbox/ymin": self._float_list(ymin),
                "image/object/bbox/xmax": self._float_list(xmax),
                "image/object/bbox/ymax": self._float_list(ymax),
                "image/object/class/text": self._bytes_list(category_names),
                "image/object/class/label": self._int64_list(category_ids),
                "image/object/is_crowd": self._int64_list(is_crowd),
                "image/object/area": self._float_list(area)
            })
            if include_masks:
                feature_dict["image/object/mask"] = self._bytes_list(encoded_mask_png)
        
        if caption_annotations:
            captions = []
            for caption_annotation in caption_annotations:
                captions.append(caption_annotation["caption"].encode("utf8"))
            feature_dict.update({
                "image/caption": self._bytes_list(captions)
            })

        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

        return key, example, num_annotations_skipped
    
    def create_category_index(self, categories):
        """Creates dictionary of COCO compatible categories keyed by category id.

        Args:
            categories: a list of dicts, each of which has the following keys:
                'id': (required) an integer id uniquely identifying this category.
                'name': (required) string representing category name
                    e.g., 'cat', 'dog', 'pizza'.

        Returns:
            category_index: a dict containing the same entries as categories, but keyed
                by the 'id' field of each category.
        """
        category_index = {}
        for cat in categories:
            category_index[cat['id']] = cat
        return category_index
    
    def _load_image_info(self, image_info_file):
        with tf.io.gfile.GFile(image_info_file, "r") as fid:
            info_dict = json.load(fid)
        
        return info_dict["images"]
    
    def _load_object_annotations(self, object_annotations_file):
        with tf.io.gfile.GFile(object_annotations_file, "r") as fid:
            object_annotations = json.load(fid)
        
        images = object_annotations["images"]
        category_index = self.create_category_index(object_annotations["categories"])

        img_to_obj_annotation = collections.defaultdict(list)
        for annotation in object_annotations["annotations"]:
            image_id = annotation["image_id"]
            img_to_obj_annotation[image_id].append(annotation)
        
        missing_annotation_count = 0
        for image in images:
            image_id = image["id"]
            if image_id not in img_to_obj_annotation:
                missing_annotation_count += 1
        logging.info("%d images are missing bboxes.", missing_annotation_count)

        return img_to_obj_annotation, category_index
    
    def create_tfrecord(self, image_dir, image_info_file, object_annotations_file, output_path, num_shards, include_masks=False):
        """Loads COCO annotation json files and convert to tfrecord format.
        
        Args:
            image_dir: Directory containing the image files.
            image_info_file: JSON file containing image info. The number of tf.Examples
                in the output tf Record files is exactly equal to the number of image info
                entries in this file. This can be any of train/val/test annotation json
                files Eg. 'image_info_test-dev2017.json',
                'instance_annotations_train2017.json',
                'caption_annotations_train2017.json', etc.
            object_annotations_file: JSON file containing bounding box annotations.
            output_path: Path to output tfrecord file.
            num_shards: Number of output files to create.     
            include_masks: Whether to include instance segmentations masks
                (PNG encoded) in the result. default: False.   
        """
        writers = [
            tf.io.TFRecordWriter(output_path + "-%05d-of-%05d.tfrecord" % (i, num_shards))
            for i in range(num_shards)
        ]

        images = self._load_image_info(image_info_file)
        img_to_obj_annotation = None
        catergory_index = None
        if object_annotations_file:
            img_to_obj_annotation, catergory_index = self._load_object_annotations(object_annotations_file)
        
        def _get_object_annotation(img_id):
            if img_to_obj_annotation:
                return img_to_obj_annotation[img_id]
            
            return None
        
        def _pool_create_tf_example(args):
            return self.create_tf_example(*args)
        
        pool = multiprocessing.Pool()
        total_num_annotations_skipped = 0
        for idx, (_, tf_example, num_annotations_skipped) in enumerate(
            pool.imap(_pool_create_tf_example, 
                [(image, image_dir, _get_object_annotation(image["id"]), catergory_index, include_masks)
                 for image in images])):
            if idx % 100 == 0:
                logging.info("On image %d of %d", idx, len(images))
            
            total_num_annotations_skipped += num_annotations_skipped
            writers[idx % num_shards].write(tf_example.SerializeToString())
        
        pool.close()
        pool.join()

        for writer in writers:
            writer.close()
        
        logging.info("Finshed writing %d images, skipped %d annotations.", len(images), total_num_annotations_skipped)



def main():
    parser = argparse.ArgumentParser("COCO tfrecord.")
    parser.add_argument("--image_dir", default=None, type=str, help="Directory containing images.")
    parser.add_argument("--image_info_file", default=None, type=str, help="File containing image information.")
    parser.add_argument("--object_annotations_file", default=None, type=str, 
                        help="File containing object annotations (boxes and instance masks).")
    parser.add_argument("--output_file_prefix", default="/tmp/train", type=str, help="Path to output file.")
    parser.add_argument("--num_shards", default=32, type=int, help="Number of shards for output file.")
    parser.add_argument("--include_masks", default=True, type=bool, 
                        help="Whether to include instance segmentations masks"
                              "(PNG encoded) in the result. default: True.")

    args = parser.parse_args()

    assert args.image_dir, "`image_dir` missing."
    assert args.image_info_file or args.object_annotations_file, "All annotation files are missing."

    if args.image_info_file:
        images_info_file = args.image_info_file
    else:
        images_info_file = args.object_annotations_file
    
    directory = os.path.dirname(args.output_file_prefix)
    if not tf.io.gfile.IsDirectory(directory):
        tf.gfile.MakeDirs(directory)
    
    dataset = COCODataset(directory).create_tfrecord(args.image_dir,
                                                     images_info_file,
                                                     args.object_annotations_file,
                                                     args.output_file_prefix,
                                                     args.num_shards,
                                                     args.include_masks)

if __name__ == "__main__":
    main()