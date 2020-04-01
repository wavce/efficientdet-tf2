import os
import cv2
import time
import glob
import random
import argparse
import numpy as np
import tensorflow as tf
from absl import app, flags, logging



def calibrating_dataset(image_dir, batch_size, input_size):
    image_names = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
    image_names = tf.convert_to_tensor(image_names)
    dataset = tf.data.Dataset.from_tensor_slices(image_names)

    def _fn(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, [input_size, input_size])

        image = tf.cast(image, tf.uint8)

        return image

    dataset = dataset.map(map_func=_fn, num_parallel_calls=8)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(1)

    return dataset


def synthetic_dataset(batch_size, input_size):
    features = np.random.normal(
      loc=112, scale=70,
      size=(batch_size, input_size, input_size, 3)).astype(np.float32)

    features = np.clip(features, 0.0, 255.0)

    features = tf.convert_to_tensor(features)
    dataset = tf.data.Dataset.from_tensor_slices([features])

    dataset = dataset.repeat()

    return dataset


def convert_to_tensorrt(mode="FP16", 
                        saved_model_dir="./efficientdet/1",
                        output_model_dir="./trt_model", 
                        use_synthetic=True, 
                        img_dir=None):
    conversion_params = tf.experimental.tensorrt.ConversionParams()
    conversion_params = conversion_params._replace(precision_mode=mode)
    conversion_params = conversion_params._replace(max_workspace_size_bytes=1<<33)
    # conversion_params = conversion_params._replace(minimum_segment_size=3)
    conversion_params = conversion_params._replace(use_calibration=mode == "INT8")
    # conversion_params = conversion_params._replace(maximum_cached_engines=1)
    conversion_params = conversion_params._replace(is_dynamic_op=True)

    converter = tf.experimental.tensorrt.Converter(input_saved_model_dir=saved_model_dir,
                                                   conversion_params=conversion_params)

    # def _input_fn(num_iterations):
    #     if not use_synthetic:
    #         dataset = calibrating_dataset(FLAGS.image_dir,
    #                                       batch_size=FLAGS.batch_size,
    #                                       input_size=FLAGS.input_size)
    #     else:
    #         dataset = synthetic_dataset(FLAGS.batch_size, FLAGS.input_size)

    #     for batch_images in dataset.take(num_iterations):

    #         yield (batch_images, )

    if conversion_params.precision_mode != "INT8":
        tf.print("  Convert to %s ..." % conversion_params.precision_mode)
        converter.convert()

        # if FLAGS.optimize_offline:
        #     tf.print("  Building TensorRT engines...")
        #     converter.build(input_fn=lambda: _input_fn(1))
        converter.save(output_model_dir)
        tf.print("  Save tensorrt model to", output_model_dir)

    # else:
    #     tf.print("  Convert to %s ..." % conversion_params.precision_mode)
    #     converter.convert(calibration_input_fn=lambda: _input_fn(FLAGS.max_num_iterations))

    #     if FLAGS.optimize_offline:
    #         tf.print("  Building TensorRT engines...")
    #         converter.build(input_fn=lambda: _input_fn(1))
    #     converter.save(FLAGS.output_trt_model)
    #     tf.print("  Save tensorrt model to", FLAGS.output_trt_model)


def main():
    parser = argparse.ArgumentParser(description="Demo args")
    parser.add_argument("--mode", default="FP16", type=str)
    parser.add_argument("--saved_model_dir", default="./saved_model/efficientdet/1", type=str)
    parser.add_argument("--output_dir", default="./trt_model/efficientdet/1", type=str)
    args = parser.parse_args()

    assert args.mode in ["FP16" or "FP32"], "Now, only support float16 and float32"
    saved_model_dir = args.saved_model_dir
    convert_to_tensorrt(args.mode, saved_model_dir, args.output_dir)


if __name__ == "__main__":
    main()
