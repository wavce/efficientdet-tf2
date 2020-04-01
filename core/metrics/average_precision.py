import tensorflow as tf


class AveragePrecision(tf.keras.metrics.Metric):
    def __init__(self, iou_threshold=0.5, use_voc07_metric=False, **kwargs):
        super(AveragePrecision, self).__init__(**kwargs)

        self.iou_threshold = iou_threshold
        self.use_voc07_metric = use_voc07_metric

        self.true_positives = []
        self.false_positives = []
        self.predicted_scores = []
        self.num_gt_boxes = 0

    def compute_tp_and_fp(self, gt_boxes, pred_boxes, pred_scores, threshold=0.5):
        num_boxes = tf.shape(pred_boxes)[0]

        fp = tf.zeros([num_boxes], tf.float32)
        tp = tf.zeros([num_boxes], tf.float32)

        pred_boxes = tf.gather(pred_boxes, tf.argsort(pred_scores, direction="DESCENDING"))
        if tf.logical_or(tf.less_equal(num_boxes, 0), tf.less_equal(tf.shape(gt_boxes)[0], 0)):
            return tp, fp

        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        matched_gt = tf.zeros([tf.shape(gt_boxes)[0]], tf.int32)
        for i in tf.range(num_boxes):
            box = pred_boxes[i]

            box_area = (box[2] - box[0]) * (box[3] - box[1])
            inter_y1 = tf.math.maximum(box[0], gt_boxes[:, 0])
            inter_x1 = tf.math.maximum(box[1], gt_boxes[:, 1])
            inter_y2 = tf.math.minimum(box[2], gt_boxes[:, 2])
            inter_x2 = tf.math.minimum(box[3], gt_boxes[:, 3])
            inter_h = tf.math.maximum(inter_y2 - inter_y1, 0)
            inter_w = tf.math.maximum(inter_x2 - inter_x1, 0)
            inter_areas = inter_h * inter_w
            union_areas = box_area + gt_areas - inter_areas
            iou = inter_areas / union_areas

            max_iou = tf.reduce_max(iou)
            arg_max_iou = tf.argmax(iou)
            # tf.print("max_iou", max_iou, "arg_max_iou", arg_max_iou, "matched_gt", matched_gt[arg_max_iou])
            if tf.logical_and(max_iou >= threshold, tf.equal(matched_gt[arg_max_iou], 0)):
                tp = tf.tensor_scatter_nd_update(tp, [[i]], [1])
                matched_gt = tf.tensor_scatter_nd_update(matched_gt, [[arg_max_iou]], [1])
            else:
                fp = tf.tensor_scatter_nd_update(fp, [[i]], [1])
      
        return tp, fp

    def compute_ap(self, precision, recall, use_voc07_metric=False):
        if use_voc07_metric:
            ap = tf.zeros([], tf.float32)
            for t in tf.range(0., 1.1, 0.1):
                mask = recall >= t
                if tf.reduce_sum(tf.cast(mask, tf.float32)) == 0:
                    p = tf.zeros([], tf.float32)
                else:
                    p = tf.reduce_max(tf.boolean_mask(precision, mask))
                ap += (p / 11.)
        else:
            # correct AP calculation
            # first append sentinel values at the end
            m_recall = tf.concat([[0.], recall, [1.]], 0)
            m_precision = tf.concat([[0.], precision, [0.]], 0)

            # compute the precision envelope
            for i in tf.range(tf.size(m_precision)-1, 0, -1):
                m_precision = tf.tensor_scatter_nd_update(
                    m_precision, [[i-1]], [tf.maximum(m_precision[i-1], m_precision[i])])
            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            indices = tf.where(m_recall[1:] != m_recall[:-1])

            # and sum (\Delta recall) * prec
            ap = tf.reduce_sum((tf.gather_nd(m_recall, indices + 1) - tf.gather_nd(m_recall, indices))
                               * tf.gather_nd(m_precision, indices + 1))
        return ap

    def update_state(self, gt_boxes, pred_boxes, pred_scores, sample_weight=None):
        for i in tf.range(tf.shape(gt_boxes)[0]):
            valid_gt_mask = tf.logical_not(tf.reduce_all(gt_boxes[i] == 0, 1))
            valid_pred_mask = tf.logical_not(tf.reduce_all(pred_boxes[i] == 0, 1))

            valid_gt_boxes = tf.boolean_mask(gt_boxes[i], valid_gt_mask)
            valid_pred_boxes = tf.boolean_mask(pred_boxes[i], valid_pred_mask)
            valid_pred_scores = tf.boolean_mask(pred_scores[i], valid_pred_mask)

            tp, fp = self.compute_tp_and_fp(gt_boxes=valid_gt_boxes, 
                                            pred_boxes=valid_pred_boxes, 
                                            pred_scores=valid_pred_scores, 
                                            threshold=self.iou_threshold)

            self.true_positives.append(tp)
            self.false_positives.append(fp)
            self.predicted_scores.append(valid_pred_scores) 
            self.num_gt_boxes += tf.shape(valid_gt_boxes)[0]

    def result(self):
        pred_scores = tf.concat(self.predicted_scores, 0)
        tp = tf.concat(self.true_positives, 0)
        fp = tf.concat(self.false_positives, 0)

        indices = tf.argsort(pred_scores, direction="DESCENDING")
        tp = tf.gather(tp, indices)
        fp = tf.gather(fp, indices)

        fp = tf.cumsum(fp)
        tp = tf.cumsum(tp)

        recall = tp / tf.cast(self.num_gt_boxes, fp.dtype)

        precision = tp / (fp + tp)
        tf.print("num_gt", self.num_gt_boxes)
        tf.print("fp", fp)
        tf.print("tp", tp)
        tf.print("recall", recall)
        tf.print("precision", precision)

        # return precision, recall, self.compute_ap(precision, recall, self.use_voc07_metric)
        return self.compute_ap(precision, recall, self.use_voc07_metric)
    
    def reset_states(self):
        del self.predicted_scores
        del self.true_positives
        del self.false_positives

        self.predicted_scores = []
        self.true_positives = []
        self.false_positives = []
        self.num_gt_boxes = 0
        
        super(AveragePrecision, self).reset_states()


def main():
    fp = [tf.constant([0, 0, 1, 0, 0, 0, 0, 1, 1, 1], tf.float32), tf.constant([1, 0, 0, 1, 1, 1], tf.float32)]
    tp = [tf.constant([1, 1, 0, 1, 1, 1, 1, 0, 0, 0], tf.float32), tf.constant([0, 1, 1, 0, 0, 0], tf.float32)]

    tp = tf.concat(tp, 0)
    fp = tf.concat(fp, 0)

    fp = tf.cumsum(fp)
    tp = tf.cumsum(tp) 
    ap = AveragePrecision().compute_ap(tp, fp)
    print(ap)

if __name__ == "__main__":
    main()
