import time
import math
import tensorflow as tf
from core import metrics
from detectors import build_detector
from data.datasets import build_dataset
from core.optimizers import build_optimizer
from core.optimizers import LookaheadOptimizer
from core.learning_rate_schedules import build_learning_rate_scheduler


def _time_to_string():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


class SingleGPUTrainer(object):
    """Train class.

        Args:
            cfg: the configuration cfg.
        """

    def __init__(self, cfg):
        use_mixed_precision = cfg.dtype == "float16"
        if use_mixed_precision:
            policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
            tf.keras.mixed_precision.experimental.set_policy(policy)

        self.train_dataset = build_dataset(name=cfg.dataset,
                                           dataset_dir=cfg.train_dataset_dir,
                                           training=True,
                                           min_level=cfg.min_level,
                                           max_level=cfg.max_level,
                                           batch_size=cfg.batch_size,
                                           input_size=cfg.input_size,
                                           augmentation=cfg.augmentation,
                                           assigner=cfg.assigner.as_dict(),
                                           anchor=cfg.anchor if cfg.anchor else None)
        self.val_dataset = build_dataset(name=cfg.dataset,
                                         dataset_dir=cfg.val_dataset_dir,
                                         training=False,
                                         min_level=cfg.min_level,
                                         max_level=cfg.max_level,
                                         batch_size=cfg.batch_size,
                                         input_size=cfg.input_size,
                                         augmentation=None,
                                         assigner=cfg.assigner.as_dict(),
                                         anchor=cfg.anchor if cfg.anchor else None)

        self.detector = build_detector(cfg.detector, cfg=cfg)

        optimizer = build_optimizer(**cfg.optimizer.as_dict())
    
        if cfg.lookahead:
            optimizer = LookaheadOptimizer(optimizer, cfg.lookahead.steps, cfg.lookahead.alpha) 

        if use_mixed_precision:
            optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                optimizer=optimizer, loss_scale= "dynamic") 

        self.optimizer = optimizer
        self.use_mixed_precision = use_mixed_precision
        self.cfg = cfg
    
        self.total_train_steps = cfg.train_steps
        self.warmup_steps = cfg.warmup.steps
        self.warmup_learning_rate = cfg.warmup.warmup_learning_rate
        self.initial_learning_rate = cfg.learning_rate_scheduler.initial_learning_rate
        self._learning_rate_scheduler = build_learning_rate_scheduler(
            **cfg.learning_rate_scheduler.as_dict(), 
            steps=self.total_train_steps, 
            warmup_steps=self.warmup_steps)

        self.global_step = tf.Variable(initial_value=0,
                                       trainable=False,
                                       name="global_step",
                                       dtype=tf.int64)

        self.val_steps = tf.Variable(0, trainable=False, name="val_steps", dtype=tf.int64)
        self.learning_rate = tf.Variable(initial_value=0,
                                         trainable=False,
                                         name="learning_rate",
                                         dtype=tf.float32)
       
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              detector=self.detector.model)
        self.manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                  directory=cfg.checkpoint_dir,
                                                  max_to_keep=10)

        latest_checkpoint = self.manager.latest_checkpoint
        if latest_checkpoint is not None:
            try:
                steps = int(latest_checkpoint.split("-")[-1])
                self.global_step.assign(steps)
            except:
                self.global_step.assign(0)
            self.checkpoint.restore(latest_checkpoint)
            tf.print(_time_to_string(), "Restored weights from %s." % latest_checkpoint)
        else:
            self.global_step.assign(0)

        self.summary_writer = tf.summary.create_file_writer(logdir=cfg.summary_dir)
        self.log_every_n_steps = cfg.log_every_n_steps
        self.save_ckpt_steps = cfg.save_ckpt_steps
        self.use_jit = tf.config.optimizer.get_jit() is not None

        self.training_loss_metrics = {}
        self.val_loss_metrics = {}
        self.ap_metric = None 
        self._add_graph = True

    def learning_rate_scheduler(self, global_step):
        with tf.name_scope("learning_rate_scheduler"):
            global_step = tf.cast(global_step, tf.float32)
            if tf.less(global_step, self.warmup_steps):
                # decayed = 0.5 * (1. - tf.math.cos(math.pi * global_step / self.warmup_steps))
                # return self.cfg.train.warmup.learning_rate * decayed
                decayed = (self.initial_learning_rate - self.warmup_learning_rate) /  self.warmup_steps
                
                return decayed * global_step + self.warmup_learning_rate 
    
            if self.cfg.train.learning_rate_scheduler.learning_rate_scheduler == "piecewise_constant":
                return self._learning_rate_scheduler(global_step)

            return self._learning_rate_scheduler(global_step - self.warmup_steps)

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, images, image_info):
        with tf.name_scope("train_step"):
            normalized_images = tf.image.convert_image_dtype(images, tf.float32)
            with tf.GradientTape(persistent=True) as tape:
                outputs = self.detector.model(normalized_images, training=True)

                loss_dict = self.detector.losses(outputs, image_info)
                loss = loss_dict["loss"] 
                if self.use_mixed_precision:
                    scaled_loss = self.optimizer.get_scaled_loss(loss)
                else:
                    scaled_loss = loss

            self.optimizer.learning_rate = self.learning_rate.value()
            gradients = tape.gradient(scaled_loss, self.detector.model.trainable_variables)
            if self.use_mixed_precision:
                gradients = self.optimizer.get_unscaled_gradients(gradients)
            
            if self.cfg.gradient_clip_norm > 0.0:
                gradients, _ = tf.clip_by_global_norm(gradients, self.cfg.train.gradient_clip_norm)
            self.optimizer.apply_gradients(zip(gradients, self.detector.model.trainable_variables))
            
            for key, value in loss_dict.items():
                if key not in self.training_loss_metrics:
                    if key == "l2_loss":
                        self.training_loss_metrics[key] = metrics.NoOpMetric()
                    else:
                        self.training_loss_metrics[key] = tf.keras.metrics.Mean()
                self.training_loss_metrics[key].update_state(value)

            if self.global_step.value() % self.log_every_n_steps == 0:
                # tf.print(self.optimizer.loss_scale._current_loss_scale, self.optimizer.loss_scale._num_good_steps)
                matched_boxes, pred_boxes, _, _ = self.detector.summary_boxes(outputs, image_info)
                images = tf.cast(images, tf.float32)
                matched_boxes = tf.cast(matched_boxes, images.dtype)
                pred_boxes = tf.cast(pred_boxes, images.dtype)
                images = tf.image.draw_bounding_boxes(images=images,
                                                    boxes=image_info["gt_boxes"] * (1. / image_info["input_size"]),
                                                    colors=tf.constant([[0., 0., 255.]]))
                images = tf.image.draw_bounding_boxes(images=images,
                                                    boxes=matched_boxes,
                                                    colors=tf.constant([[255., 0., 0.]]))  
                images = tf.image.draw_bounding_boxes(images=images,
                                                    boxes=pred_boxes,
                                                    colors=tf.constant([[0., 255., 0.]]))    
                images = tf.cast(images, tf.uint8)
                with tf.device("/cpu:0"):
                    with self.summary_writer.as_default():
                        tf.summary.image("train/images", images, self.global_step, 5)

            return loss

    @tf.function(experimental_relax_shapes=True)
    def val_step(self, images, image_info):
        with tf.name_scope("val_step"):
            normalized_images = tf.image.convert_image_dtype(images, tf.float32)
            outputs = self.detector.model(normalized_images, training=False)
            loss_dict = self.detector.losses(outputs, image_info)

            for key, value in loss_dict.items():
                if key not in self.val_loss_metrics:
                    if key == "l2_loss":
                        self.val_loss_metrics[key] = metrics.NoOpMetric()
                    else:
                        self.val_loss_metrics[key] = tf.keras.metrics.Mean()
                
                self.val_loss_metrics[key].update_state(value)
            
            matched_boxes, pred_boxes, pred_scores, pred_classes = self.detector.summary_boxes(outputs, image_info)
            matched_boxes = tf.cast(matched_boxes, images.dtype)
            pred_boxes = tf.cast(pred_boxes, images.dtype)
            gt_boxes = image_info["gt_boxes"]
            if self.val_steps.value() % self.log_every_n_steps == 0:
                images = tf.cast(images, tf.float32)
                images = tf.image.draw_bounding_boxes(images=images,
                                                    boxes=gt_boxes * (1. / image_info["input_size"]),
                                                    colors=tf.constant([[0., 0., 255.]]))
                images = tf.image.draw_bounding_boxes(images=images,
                                                    boxes=matched_boxes,
                                                    colors=tf.constant([[255., 0., 0.]]))
                images = tf.image.draw_bounding_boxes(images=images,
                                                    boxes=pred_boxes,
                                                    colors=tf.constant([[0., 255., 0.]]))       
                images = tf.cast(images, tf.uint8)
                with tf.device("/cpu:0"):
                    with self.summary_writer.as_default():
                        tf.summary.image("val/images", images, self.val_steps.value(), 5)
            
            return gt_boxes, image_info["gt_labels"], pred_boxes * image_info["input_size"], pred_scores, pred_classes + 1

    def run(self):
        count = 0
        self.ap_metric = metrics.mAP(self.cfg.num_classes)
        max_ap = 0
        # TRAIN LOOP
        start = time.time()
        for images, image_info in self.train_dataset.take(self.total_train_steps):
            self.global_step.assign_add(1)
            lr = self.learning_rate_scheduler(self.global_step.value())
            self.learning_rate.assign(lr)
            count += 1

            if self._add_graph:
                tf.summary.trace_on(graph=True, profiler=True)
                self.train_step(images, image_info)
                with self.summary_writer.as_default():
                    tf.summary.trace_export(name=self.cfg.detector,
                                            step=0,
                                            profiler_outdir=self.cfg.summary_dir)
                self._add_graph = False
            else:
                self.train_step(images, image_info)

            info = [_time_to_string(), "TRAINING", self.global_step]
            if tf.equal(self.global_step % self.log_every_n_steps, 0):
                with self.summary_writer.as_default():
                    for key in self.training_loss_metrics:
                        value = self.training_loss_metrics[key].result()
                        self.training_loss_metrics[key].reset_states()
                        tf.summary.scalar("train/" + key, value, self.global_step)
                        info.extend([key, "=", value])

                    tf.summary.scalar("learning_rate", self.learning_rate.value(), self.global_step)
                    info.extend(["lr", "=", self.learning_rate.value()])
                info.append("(%.2fs)" % ((time.time() - start) / count))
                tf.print(*info)
                start = time.time()
                count = 0
            
            if self.global_step >= self.total_train_steps:
                    tf.print(_time_to_string(), "Train over.")
                    break
            
            if tf.logical_and(self.global_step % self.cfg.val_every_n_steps == 0, 
                              self.global_step > self.total_train_steps // 3):
                # VAL LOOP
                tf.print("=" * 150)
                val_start = time.time()
                
                for images, image_info in self.val_dataset:
                    self.val_steps.assign_add(1)
                    gt_boxes, gt_labels, pred_boxes, pred_scores, pred_classes = self.val_step(images, image_info)
                    self.ap_metric.update_state(gt_boxes,
                                                gt_labels,
                                                pred_boxes, 
                                                pred_scores,
                                                pred_classes)
                
                info = [_time_to_string(), "VAL", self.global_step]
                with self.summary_writer.as_default():
                    for key in self.val_loss_metrics:
                        result = self.val_loss_metrics[key].result()
                        self.val_loss_metrics[key].reset_states()
                        tf.summary.scalar("val/" + key, result, self.global_step)
                        info.extend([key, "=", result])
                    ap = self.ap_metric.result()
                    tf.summary.scalar("val/ap", ap, step=self.global_step)
                    info.extend(["ap =", ap])
                    self.ap_metric.reset_states()
                val_end = time.time()
                info.extend(["(%.2fs)" % (val_end - val_start)])
                tf.print(*info)

                if ap > max_ap:
                    self.manager.save(self.global_step)
                    tf.print(_time_to_string(), "Saving detector to %s." % self.manager.latest_checkpoint)
                    max_ap = ap
            else:
                if tf.equal(self.global_step % self.save_ckpt_steps, 0):
                    tf.print("=" * 150)
                    self.manager.save(self.global_step)
                    tf.print(_time_to_string(), "Saving detector to %s." % self.manager.latest_checkpoint)
                    tf.print("=" * 150)

        self.summary_writer.close()
        tf.print(_time_to_string(), "Training over.")
