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


class MultiGPUTrainer(object):
    """Train class.

        Args:
            detector: detector.
            strategy: Distribution strategy in use.
            cfg: the configuration cfg.
        """

    def __init__(self, cfg):
        strategy = tf.distribute.MirroredStrategy()
        self.num_replicas = strategy.num_replicas_in_sync
        self.train_batch_size = self.num_replicas * cfg.train.dataset.batch_size 
        self.val_batch_size = self.num_replicas * cfg.val.dataset.batch_size

        train_dataset = build_dataset(dataset=cfg.train.dataset.dataset,
                                      dataset_dir=cfg.train.dataset.dataset_dir,
                                      batch_size=self.train_batch_size,
                                      training=cfg.train.dataset.training,
                                      input_size=cfg.train.dataset.input_size,
                                      augmentation=cfg.train.dataset.augmentation,
                                      assigner=cfg.assigner.as_dict(),
                                      anchor=cfg.anchor.as_dict() if cfg.anchor else None)
        val_dataset = build_dataset(dataset=cfg.val.dataset.dataset,
                                    dataset_dir=cfg.val.dataset.dataset_dir,
                                    batch_size=self.val_batch_size,
                                    training=cfg.val.dataset.training,
                                    input_size=cfg.val.dataset.input_size,
                                    augmentation=cfg.val.dataset.augmentation,
                                    assigner=cfg.assigner.as_dict(),
                                    anchor=cfg.anchor.as_dict() if cfg.anchor else None)

        with strategy.scope():
            self.train_dataset = strategy.experimental_distribute_dataset(train_dataset)
            self.val_dataset = strategy.experimental_distribute_dataset(val_dataset)

            use_mixed_precision = cfg.dtype == "float16"
            if use_mixed_precision:
                policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
                tf.keras.mixed_precision.experimental.set_policy(policy)

            self.detector = build_detector(cfg.detector, cfg=cfg)

            optimizer = build_optimizer(**cfg.train.optimizer.as_dict())
            tf.print(_time_to_string(), "The optimizers is %s." % cfg.train.optimizer.optimizer)

            if cfg.train.lookahead:
                tf.print(_time_to_string(), "Using Lookahead Optimizer.")
                optimizer = LookaheadOptimizer(optimizer, cfg.train.lookahead.steps, cfg.train.lookahead.alpha)

            if use_mixed_precision:
                loss_scale = (cfg.train.mixed_precision.loss_scale
                              if cfg.train.mixed_precision.loss_scale is not None and
                                  cfg.train.mixed_precision.loss_scale > 0
                              else "dynamic")
                optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                    optimizer=optimizer, loss_scale=loss_scale)
                tf.print(_time_to_string(), "Using mixed precision training, loss scale is {}.".format(loss_scale))
        
            self.optimizer = optimizer
            self.strategy = strategy
            self.use_mixed_precision = use_mixed_precision
            self.cfg = cfg

            self.total_train_steps = cfg.train.train_steps
            self.warmup_steps = cfg.train.warmup.steps
            self.warmup_learning_rate = cfg.train.warmup.warmup_learning_rate
            self.initial_learning_rate = cfg.train.learning_rate_scheduler.initial_learning_rate
            self._learning_rate_scheduler = build_learning_rate_scheduler(
                **cfg.train.learning_rate_scheduler.as_dict(), 
                steps=self.total_train_steps, 
                warmup_steps=self.warmup_steps)
        
             
            tf.print(_time_to_string(),
                    "The leaning rate scheduler is %s, initial learning rate is %f." % (
                        cfg.train.learning_rate_scheduler.learning_rate_scheduler, 
                        cfg.train.learning_rate_scheduler.initial_learning_rate))
            if self.warmup_steps > 0:
                tf.print(_time_to_string(), "Using warm-up learning rate policy, "
                        "warm-up learning rate %f, training %d steps." % (self.warmup_learning_rate, self.warmup_steps))
            tf.print(_time_to_string(), "Training", self.total_train_steps, 
                    "steps, every step has", self.train_batch_size, "images.")
            self.global_step = tf.Variable(initial_value=0,
                                           trainable=False,
                                           name="global_step",
                                           dtype=tf.int64)

            self.val_steps = tf.Variable(0, trainable=False, name="val_steps", dtype=tf.int64)
            self.learning_rate = tf.Variable(initial_value=0,
                                             trainable=False,
                                             name="learning_rate",
                                             dtype=tf.float32)
            self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, detector=self.detector.model)
            self.manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                      directory=cfg.train.checkpoint_dir,
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

            self.summary_writer = tf.summary.create_file_writer(logdir=cfg.train.summary_dir)
            self.log_every_n_steps = cfg.train.log_every_n_steps
            self.save_ckpt_steps = cfg.train.save_ckpt_steps
            self.use_jit = tf.config.optimizer.get_jit() is not None

            self.training_loss_metrics = {}
            self.val_loss_metrics = {}
            self.ap_metric = None 
            self._add_graph = True

    def run(self):
        with self.strategy.scope():
            # `experimental_run_v2` replicates the provided computation and runs it
            # with the distributed input.
            @tf.function(experimental_relax_shapes=True, input_signature=self.train_dataset.element_spec)
            def distributed_train_step(images, labels):
                # @tf.function(experimental_relax_shapes=True)
                def step_fn(batch_images, batch_labels):
                    normalized_images = tf.image.convert_image_dtype(batch_images, tf.float32)
                    with tf.GradientTape() as tape:
                        outputs = self.detector.model(normalized_images, training=True)
                        loss_dict = self.detector.losses(outputs, batch_labels)
                        
                        loss = loss_dict["loss"] * (1. / self.num_replicas)
                        if self.use_mixed_precision:
                            scaled_loss = self.optimizer.get_scaled_loss(loss)
                        else:
                            scaled_loss = loss

                    self.optimizer.learning_rate = self.learning_rate.value()
                    gradients = tape.gradient(scaled_loss, self.detector.model.trainable_variables)
                    if self.use_mixed_precision:
                        gradients = self.optimizer.get_unscaled_gradients(gradients)
                    self.optimizer.apply_gradients(zip(gradients, self.detector.model.trainable_variables))

                    for key, value in loss_dict.items():
                        if key not in self.training_loss_metrics:
                            if key == "l2_loss":
                                self.training_loss_metrics[key] = metrics.NoOpMetric()
                            else:
                                self.training_loss_metrics[key] = tf.keras.metrics.Mean()
                        
                        if key == "l2_loss":
                            self.training_loss_metrics[key].update_state(value / self.num_replicas) 
                        else:
                            self.training_loss_metrics[key].update_state(value)

                    if self.global_step.value() % self.log_every_n_steps == 0:
                        # tf.print(self.optimizer.loss_scale._current_loss_scale, self.optimizer.loss_scale._num_good_steps)
                        matched_boxes, nmsed_boxes, _ = self.detector.summary_boxes(outputs, batch_labels)
                        batch_gt_boxes = batch_labels["gt_boxes"] * (1. / batch_labels["input_size"]) 
                        batch_images = tf.image.draw_bounding_boxes(images=batch_images,
                                                                    boxes=batch_gt_boxes,
                                                                    colors=tf.constant([[0., 0., 255.]]))
                        batch_images = tf.image.draw_bounding_boxes(images=batch_images,
                                                                    boxes=matched_boxes,
                                                                    colors=tf.constant([[255., 0., 0.]]))
                        batch_images = tf.image.draw_bounding_boxes(images=batch_images,
                                                                    boxes=nmsed_boxes,
                                                                    colors=tf.constant([[0., 255., 0.]]))
                        batch_images = tf.cast(batch_images, tf.uint8)
                        with tf.device("/cpu:0"):
                            with self.summary_writer.as_default():
                                tf.summary.image("train/images", batch_images, self.global_step, 5)

                    return loss
                
                per_replica_losses = self.strategy.experimental_run_v2(step_fn, args=(images, labels))
                return self.strategy.reduce(tf.distribute.ReduceOp.SUM,
                                            per_replica_losses,
                                            axis=None)

            @tf.function(experimental_relax_shapes=True, input_signature=self.train_dataset.element_spec)
            def distributed_valuate_step(images, labels):
                def step_fn(batch_images, batch_labels):
                    normalized_images = tf.image.convert_image_dtype(batch_images, tf.float32)
                    outputs = self.detector.model(normalized_images, training=False)
                    loss_dict = self.detector.losses(outputs, batch_labels)

                    for key, value in loss_dict.items():
                        if key not in self.val_loss_metrics:
                            if key == "l2_loss":
                                self.val_loss_metrics[key] = metrics.NoOpMetric()
                            else:
                                self.val_loss_metrics[key] = tf.keras.metrics.Mean()

                        if key == "l2_loss":
                            self.val_loss_metrics[key].update_state(value / self.num_replicas)
                        else:
                            self.val_loss_metrics[key].update_state(value)
        
                    matched_boxes, nmsed_boxes, nmsed_scores = self.detector.summary_boxes(outputs, batch_labels)
                    batch_gt_boxes = batch_labels["gt_boxes"] * (1. / batch_labels["input_size"]) 
                    if self.val_steps.value() % self.log_every_n_steps == 0:
                        batch_images = tf.image.draw_bounding_boxes(images=batch_images,
                                                                    boxes=batch_gt_boxes,
                                                                    colors=tf.constant([[0., 0., 255.]]))
                        batch_images = tf.image.draw_bounding_boxes(images=batch_images,
                                                                    boxes=matched_boxes,
                                                                    colors=tf.constant([[255., 0., 0.]]))
                        batch_images = tf.image.draw_bounding_boxes(images=batch_images,
                                                                    boxes=nmsed_boxes,
                                                                    colors=tf.constant([[0., 255., 0.]]))
                        batch_images = tf.cast(batch_images, tf.uint8)
                        
                        with tf.device("/cpu:0"):
                            with self.summary_writer.as_default():
                                tf.summary.image("valuate/images", batch_images, self.val_steps.value(), 5)

                    return batch_gt_boxes, nmsed_boxes, nmsed_scores

                return self.strategy.experimental_run_v2(step_fn, args=(images, labels))
            
            def learning_rate_scheduler(global_step):
                global_step = tf.cast(global_step, tf.float32)
                if tf.less(global_step, self.warmup_steps):
                    # decayed = 0.5 * (1. - tf.math.cos(math.pi * global_step / self.warmup_steps))
                    # return self.cfg.train.warmup.learning_rate * decayed

                    return ((self.initial_learning_rate - self.warmup_learning_rate) 
                            * global_step / self.warmup_steps + self.warmup_learning_rate)
                    
                if self.cfg.train.learning_rate_scheduler.learning_rate_scheduler == "piecewise_constant":
                    return self._learning_rate_scheduler(global_step)

                return self._learning_rate_scheduler(global_step - self.warmup_steps)

            count = 0
            max_ap = 0
            self.ap_metric = metrics.AveragePrecision()
            # TRAIN LOOP
            start = time.time()
            for images, image_info in self.train_dataset:
                self.global_step.assign_add(1)
                lr = learning_rate_scheduler(self.global_step.value())
                self.learning_rate.assign(lr)
                if self._add_graph:
                    tf.summary.trace_on(graph=True, profiler=True)
                    distributed_train_step(images, image_info)

                    with self.summary_writer.as_default():
                        tf.summary.trace_export(name=self.cfg.detector,
                                                step=0,
                                                profiler_outdir=self.cfg.train.summary_dir)
                    self._add_graph = False
                else:
                    distributed_train_step(images, image_info)

                count += 1

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

                if tf.logical_and(self.global_step % self.cfg.val.val_every_n_steps == 0, 
                                self.global_step > self.total_train_steps // 3):
                    tf.print("=" * 150)
                    # EVALUATING LOO
                    val_start = time.time()
                    for images, image_info in self.val_dataset:
                        self.val_steps.assign_add(1)
                        gt_boxes, pred_boxes, pred_scores = distributed_valuate_step(images, image_info)

                        gt_boxes = [b for x in tf.nest.flatten(gt_boxes) for b in self.strategy.unwrap(x)]
                        pred_boxes = [b for x in tf.nest.flatten(pred_boxes) for b in self.strategy.unwrap(x)]
                        pred_scores = [s for x in tf.nest.flatten(pred_scores) for s in self.strategy.unwrap(x)]
                        gt_boxes = tf.concat(gt_boxes, 0)

                        pred_boxes = tf.concat(pred_boxes, 0)
                        pred_scores = tf.concat(pred_scores, 0)

                        self.ap_metric.update_state(gt_boxes, pred_boxes, pred_scores)

                    val_end = time.time()
                    template = [_time_to_string(), "EVALUATING", self.global_step]
                    with self.summary_writer.as_default():
                        for name in self.val_loss_metrics:
                            value = self.val_loss_metrics[name].result()
                            tf.summary.scalar("val/" + name, value, self.global_step)
                            template.extend([name, "=", value])
                            self.val_loss_metrics[name].reset_states()
                    
                        ap = self.ap_metric.result()
                        self.ap_metric.reset_states()
                        tf.summary.scalar("val/ap", ap, self.global_step)
                        template.extend(["ap =", ap, "(%.2fs)." % (val_end - val_start)])
                    tf.print(*template)

                    if ap > max_ap:
                        self.manager.save(self.global_step)
                        tf.print(_time_to_string(), "Saving detector to %s." % self.manager.latest_checkpoint)
                        max_ap = ap
                        start = time.time()
                        count = 0
                else:
                    if tf.equal(self.global_step % self.save_ckpt_steps, 0):
                        self.manager.save(self.global_step)
                        tf.print(_time_to_string(), "Saving detector to %s." % self.manager.latest_checkpoint)

        self.manager.save(self.global_step)
        self.summary_writer.close()
