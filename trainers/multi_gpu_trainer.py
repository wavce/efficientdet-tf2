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
        self.train_batch_size = self.num_replicas * cfg.batch_size 
        self.val_batch_size = self.num_replicas * cfg.batch_size

        train_dataset = build_dataset(name=cfg.dataset,
                                      dataset_dir=cfg.train_dataset_dir,
                                      training=True,
                                      min_level=cfg.min_level,
                                      max_level=cfg.max_level,
                                      batch_size=self.train_batch_size,
                                      input_size=cfg.input_size,
                                      augmentation=cfg.augmentation,
                                      assigner=cfg.assigner.as_dict(),
                                      anchor=cfg.anchor if cfg.anchor else None)
        val_dataset = build_dataset(name=cfg.dataset,
                                    dataset_dir=cfg.val_dataset_dir,
                                    training=False,
                                    min_level=cfg.min_level,
                                    max_level=cfg.max_level,
                                    batch_size=self.val_batch_size,
                                    input_size=cfg.input_size,
                                    augmentation=None,
                                    assigner=cfg.assigner.as_dict(),
                                    anchor=cfg.anchor if cfg.anchor else None) 

        with strategy.scope():
            self.train_dataset = strategy.experimental_distribute_dataset(train_dataset)
            self.val_dataset = strategy.experimental_distribute_dataset(val_dataset)

            use_mixed_precision = cfg.dtype == "float16"
            if use_mixed_precision:
                policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
                tf.keras.mixed_precision.experimental.set_policy(policy)

            self.detector = build_detector(cfg.detector, cfg=cfg)

            optimizer = build_optimizer(**cfg.optimizer.as_dict())
    
            if cfg.lookahead:
                optimizer = LookaheadOptimizer(optimizer, cfg.lookahead.steps, cfg.lookahead.alpha) 

            if use_mixed_precision:
                optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                    optimizer=optimizer, loss_scale= "dynamic") 
        
            self.optimizer = optimizer
            self.strategy = strategy
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
            self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, detector=self.detector.model)
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

    def run(self):
        with self.strategy.scope():
            # `experimental_run_v2` replicates the provided computation and runs it
            # with the distributed input.
            @tf.function(experimental_relax_shapes=True, input_signature=self.train_dataset.element_spec)
            def distributed_train_step(images, labels):
                # @tf.function(experimental_relax_shapes=True)
                def step_fn(batch_images, batch_labels):
                    with tf.name_scope("train_step"):
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
                            matched_boxes, nmsed_boxes, _, _ = self.detector.summary_boxes(outputs, batch_labels)
                            batch_gt_boxes = batch_labels["gt_boxes"] * (1. / batch_labels["input_size"]) 
                            batch_images = tf.cast(batch_images, tf.float32)
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
                
                per_replica_losses = self.strategy.run(step_fn, args=(images, labels))
                return self.strategy.reduce(tf.distribute.ReduceOp.SUM,
                                            per_replica_losses,
                                            axis=None)

            @tf.function(experimental_relax_shapes=True, input_signature=self.train_dataset.element_spec)
            def distributed_valuate_step(images, labels):
                def step_fn(batch_images, batch_labels):
                    with tf.name_scope("val_step"):
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
            
                        matched_boxes, nmsed_boxes, nmsed_scores, nmsed_classes = self.detector.summary_boxes(outputs, batch_labels)
                        batch_gt_boxes = batch_labels["gt_boxes"]  
                        if self.val_steps.value() % self.log_every_n_steps == 0:
                            batch_images = tf.cast(batch_images, tf.float32)
                            batch_images = tf.image.draw_bounding_boxes(images=batch_images,
                                                                        boxes=batch_gt_boxes * (1. / batch_labels["input_size"]),
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

                        return (batch_gt_boxes, batch_labels["gt_labels"],
                                nmsed_boxes * batch_labels["input_size"], 
                                nmsed_scores, nmsed_classes + 1)

                return self.strategy.run(step_fn, args=(images, labels))
            
            def learning_rate_scheduler(global_step):
                with tf.name_scope("learning_rate_scheduler"):
                    global_step = tf.cast(global_step, tf.float32)
                    if tf.less(global_step, self.warmup_steps):

                        return ((self.initial_learning_rate - self.warmup_learning_rate) 
                                * global_step / self.warmup_steps + self.warmup_learning_rate)
                        
                    if self.cfg.learning_rate_scheduler.scheduler == "piecewise_constant":
                        return self._learning_rate_scheduler(global_step)

                    return self._learning_rate_scheduler(global_step - self.warmup_steps)

            count = 0
            max_ap = 0
            self.ap_metric = metrics.mAP(self.cfg.num_classes)
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
                                                profiler_outdir=self.cfg.summary_dir)
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

                if tf.logical_and(self.global_step % self.cfg.val_every_n_steps == 0, 
                                  self.global_step > self.total_train_steps // 3):
                    tf.print("=" * 150)
                    # EVALUATING LOO
                    val_start = time.time()
                    for images, image_info in self.val_dataset:
                        self.val_steps.assign_add(1)
                        gt_boxes, gt_labels, pred_boxes, pred_scores, pred_classes = distributed_valuate_step(
                            images, image_info)

                        gt_boxes = [b for x in tf.nest.flatten(gt_boxes) for b in self.strategy.unwrap(x)]
                        gt_labels = [l for x in tf.nest.flatten(gt_labels) for l in self.strategy.unwrap(x)]
                        pred_boxes = [b for x in tf.nest.flatten(pred_boxes) for b in self.strategy.unwrap(x)]
                        pred_scores = [s for x in tf.nest.flatten(pred_scores) for s in self.strategy.unwrap(x)]
                        pred_classes = [c for x in tf.nest.flatten(pred_classes) for c in self.strategy.unwrap(x)]
                        gt_boxes = tf.concat(gt_boxes, 0)
                        gt_labels = tf.concat(gt_labels, 0)
                        pred_boxes = tf.concat(pred_boxes, 0)
                        pred_scores = tf.concat(pred_scores, 0)
                        pred_classes = tf.concat(pred_classes, 0)

                        self.ap_metric.update_state(gt_boxes,
                                                    gt_labels,
                                                    pred_boxes, 
                                                    pred_scores,
                                                    pred_classes)

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
                        tf.summary.scalar("val/ap", ap[0], self.global_step)
                        template.extend(["ap =", ap, "(%.2fs)." % (val_end - val_start)])
                    tf.print(*template)

                    if tf.logical_or(ap > max_ap, ap - max_ap > -1.):
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
