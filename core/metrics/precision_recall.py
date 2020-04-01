import tensorflow as tf


class PrecisionRecall(tf.keras.metrics.Metric):
    def __init__(self, iou_threshold=0.5, name="precision_recall", **kwargs):
      super(PrecisionRecall, self).__init__(name=name, **kwargs)

      self.iou_threshold = iou_threshold
      self.true_positives = self.add_weight(name="tp", initializer="zeros")
      self.false_negatives = self.add_weight(name="fp", initializer="zeros")
      self.num_gt_boxes = self.add_weight(name="num_gt_boxes", initializer="zeros")             
    
    def _update_tp_fp_per_image(self, ground_truth_boxes, predicted_boxes):
        with tf.name_scope("update_tp_fp_per_image"):
            ground_truth_boxes = tf.boolean_mask(ground_truth_boxes, 
                                                 tf.logical_not(tf.reduce_all(ground_truth_boxes == 0, 1)))
            predicted_boxes = tf.boolean_mask(predicted_boxes,
                                              tf.logical_not(tf.reduce_all(predicted_boxes == 0, 1)))

            num_gt_boxes = tf.shape(ground_truth_boxes)[0]
            num_pred_boxes = tf.shape(predicted_boxes)[[0]]
            matched_gt_boxes = tf.zeros([tf.shape(ground_truth_boxes)[0]], dtype=tf.int32)
            self.num_gt_boxes.assign_add(tf.cast(num_gt_boxes, self.num_gt_boxes.dtype))
            
            gt_areas = ((ground_truth_boxes[:, 2] - ground_truth_boxes[:, 0]) *
                        (ground_truth_boxes[:, 3] - ground_truth_boxes[:, 1]))
            pred_areas = ((predicted_boxes[:, 2] - predicted_boxes[:, 0]) * 
                          (predicted_boxes[:, 3] - predicted_boxes[:, 1]))

            if tf.greater(tf.shape(predicted_boxes)[0], 0):
                for i in tf.range(num_pred_boxes):
                    box = predicted_boxes[i]
                    inter_y1 = tf.math.maximum(box[0], ground_truth_boxes[:, 0])
                    inter_x1 = tf.math.maximum(box[1], ground_truth_boxes[:, 1])
                    inter_y2 = tf.math.minimum(box[2], ground_truth_boxes[:, 2])
                    inter_x2 = tf.math.minimum(box[3], ground_truth_boxes[:, 3])

                    inter_h = tf.math.maximum(inter_y2 - inter_y1, 0.0)
                    inter_w = tf.math.maximum(inter_x2 - inter_x1, 0.0)
                    inter_areas = inter_h * inter_w
                    
                    iou = inter_areas / (gt_areas + pred_areas[i] - inter_areas)
                
                    max_iou = tf.reduce_max(iou)
                    
                    if tf.greater_equal(max_iou, self.iou_threshold):
                        arg_max_iou = tf.argmax(iou)
                        if tf.not_equal(matched_gt_boxes[arg_max_iou], 1):
                            self.true_positives.assign_add(1)
                            matched_gt_boxes = tf.tensor_scatter_nd_update(matched_gt_boxes, 
                                                                           tf.reshape(arg_max_iou, [1, 1]), 
                                                                           tf.constant([1], dtype=matched_gt_boxes.dtype))
                        else:
                            self.false_negatives.assign_add(1)
                    else:
                        self.false_negatives.assign_add(1)

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_size = tf.shape(y_true)[0]
        for i in tf.range(batch_size):
            if tf.greater(tf.shape(y_true[i])[0], 0):
                self._update_tp_fp_per_image(y_true[i], y_pred[i])

    def result(self):
        precision = tf.cond(self.true_positives > 0,
                            lambda: self.true_positives / (self.true_positives + self.false_negatives),
                            lambda: self.true_positives)
        recall = self.true_positives / self.num_gt_boxes

        return precision, recall
