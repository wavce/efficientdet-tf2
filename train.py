import os
from absl import app
from absl import flags
import tensorflow as tf
from absl import logging
from configs import build_configs
from trainers import MultiGPUTrainer
from trainers import SingleGPUTrainer


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FLAGS = flags.FLAGS
flags.DEFINE_string("name",
                    default="efficientdet",
                    help="The detector name, e.g. "
                         " `efficientdet`, `efficient_fcos`.")
flags.DEFINE_bool("multi_gpu_training",
                  default=False,
                  help="Use multi-gpu training or not,"
                       " default False, means use one gpu.")

FLAGS.mark_as_parsed()


def main(_):
    # logger = tf.get_logger()
    # logger.setLevel(logging.DEBUG)

    # tf.random.set_seed(2333)
    tf.config.optimizer.set_jit(True)
    # tf.config.optimizer.set_experimental_options()
    # tf.debugging.enable_check_numerics()

    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    if FLAGS.multi_gpu_training:
        trainer = MultiGPUTrainer(build_configs(FLAGS.name))
    else:
        trainer = SingleGPUTrainer(build_configs(FLAGS.name))
    
    trainer.run()


if __name__ == '__main__':
    app.run(main)
