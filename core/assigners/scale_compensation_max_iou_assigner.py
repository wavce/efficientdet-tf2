import tensorflow as tf
from core.assigners import MaxIoUAssigner


class ScaleCompensationMaxIoUAssigner(MaxIoUAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index:
        - -1: don't care
        - 0: negative sample, no assigned gt
        - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        pos_iou_thresh (float): IoU threshold for positive boxes.
        neg_iou_thresh (float or tuple): IoU threshold for negative boxes.
    """
    def __init__(self,
                 pos_iou_thresh,
                 neg_iou_thresh=None,
                 **kwargs):
        super(ScaleCompensationMaxIoUAssigner, self).__init__(
            pos_iou_thresh=pos_iou_thresh,
            neg_iou_thresh=neg_iou_thresh,
            **kwargs)

    def assign_wrt_overlaps(self, overlaps, gt_boxes):
        """The assignment is done in following steps, the order matters:
        1. assign every box to -1.
        2. assign proposals whose iou with all gts < neg_iou_thresh to  0.
        3. for each box, if the iou with its nearest gt >= pos_iou_thresh,
            assign it to that box.
        4. for each gt box, assign its nearest proposals (may be more than
            one) to itself.

        Args:
            overlaps (Tensor): Bounding boxes to be assigned, shape (n, k).
            gt_boxes (Tensor): Ground-truth boxes, shape (k, 4).

        Returns:
            target_boxes (Tensor), target_labels (Tensor).
        """
        num_gts = tf.shape(overlaps)[1]
        num_proposals = tf.shape(overlaps)[0]

        # 1. assign every proposal to -1
        target_labels = tf.fill([num_proposals], value=-1)

        max_overlaps = tf.reduce_max(overlaps, 1)
        argmax_overlaps = tf.argmax(overlaps, 1)

        target_boxes = tf.gather(gt_boxes, argmax_overlaps)

        positive = tf.greater_equal(max_overlaps, self.pos_iou_thresh)
        target_labels = tf.where(positive, tf.ones_like(target_labels), target_labels)

        num_pos = tf.reduce_sum(tf.cast(positive, gt_boxes.dtype))

        gt_max_overlaps = tf.reduce_max(overlaps, 0)
        matched_mask = tf.cast(tf.greater_equal(overlaps, self.pos_iou_thresh), gt_boxes.dtype)
        num_matched = tf.reduce_sum(matched_mask, axis=0)

        matched_gt_mask = tf.greater_equal(gt_max_overlaps, self.pos_iou_thresh)
        num_matched_gts = tf.reduce_sum(tf.cast(matched_gt_mask, gt_boxes.dtype))

        average_matched_anchor = tf.cond(tf.greater(num_matched_gts, 0),
                                         lambda: num_pos / num_matched_gts,
                                         lambda: tf.constant(10., dtype=gt_boxes.dtype))

        # 2. assign negative
        negative = tf.logical_and(tf.greater_equal(max_overlaps, self.neg_iou_thresh[0]),
                                  tf.less(max_overlaps, self.neg_iou_thresh[1]))
        target_labels = tf.where(negative, tf.zeros_like(target_labels), target_labels)

        # 3.
        for i in tf.range(num_gts):
            box = gt_boxes[i:i+1]
            if tf.reduce_all(box == 0):
                continue
            
            label = tf.constant([1], dtype=target_labels.dtype)
            ind = tf.constant([1], dtype=indicator.dtype)
            if tf.logical_and(tf.less(num_matched[i], average_matched_anchor), tf.greater(num_pos, 0.)):

                current_overlaps = overlaps[:, i]
                k = tf.cast(average_matched_anchor, tf.int32)
                topk_overlaps, indices = tf.nn.top_k(current_overlaps, k=k)

                indices = tf.boolean_mask(indices, tf.greater_equal(topk_overlaps, 0.1))
                indices = tf.expand_dims(indices, 1)
                k = tf.shape(indices)[0]

                boxes = tf.tile(box, [k, 1])
                labels = tf.tile(label, [k])
                inds = tf.tile(ind, [k])
                target_boxes = tf.tensor_scatter_nd_update(target_boxes, indices, boxes)
                target_labels = tf.tensor_scatter_nd_update(target_labels, indices, labels)
            # else:
            #     max_overlaps_inds = tf.where(tf.greater_equal(overlaps[:, i], gt_max_overlaps[i]))
            #
            #     boxes = tf.tile(box, [tf.shape(max_overlaps_inds)[0], 1])
            #     labels = tf.tile(label, [tf.shape(max_overlaps_inds)[0]])
            #     target_boxes = tf.tensor_scatter_nd_update(target_boxes, max_overlaps_inds, boxes)
            #     target_labels = tf.tensor_scatter_nd_update(target_labels, max_overlaps_inds, labels)

        return target_boxes, target_labels

