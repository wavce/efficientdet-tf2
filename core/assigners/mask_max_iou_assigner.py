import tensorflow as tf
from core.bbox import unaligned_box_iou


class MaskMaxIoUAssigner(object):
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
                 num_classes=2):
        neg_iou_thresh = neg_iou_thresh if neg_iou_thresh is not None else pos_iou_thresh
        if isinstance(neg_iou_thresh, (int, float)):
            neg_iou_thresh = [0., neg_iou_thresh]

        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh =neg_iou_thresh
        self.num_classes = num_classes

    def assign(self, gt_boxes, gt_labels, proposals, feat_size, strides):
        """Assign gt to boxes/

        This method assign a gt box to every box (proposal/anchor), each box
        will be assigned with -1, 0 or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.

        The assignment is done in following steps, the order matters:
        1. assign every box to -1.
        2. assign proposals whose iou with all gts < neg_iou_thresh to  0.
        3. for each box, if the iou with its nearest gt >= pos_iou_thresh,
            assign it to that box.
        4. for each gt box, assign its nearest proposals (may be more than
            one) to itself.

        Args:
            proposals (Tensor): Bounding boxes to be assigned, shape (n, 4).
            gt_boxes (Tensor): Ground-truth boxes, shape (k, 4).
            gt_labels (Tensor): Ground-truth labels, shape (k, ).
            feat_size (Tensor): The feature map size.

        Returns:
            target_boxes (Tensor), target_labels (Tensor).
        """
        gt_boxes = tf.concat([gt_boxes, tf.zeros([1, 4], dtype=gt_boxes.dtype)], axis=0)[:200]
        gt_labels = tf.concat([gt_labels, tf.zeros([1], dtype=gt_labels.dtype)], axis=0)[:200] 

        overlaps = unaligned_box_iou(proposals, gt_boxes)

        # if tf.greater(self.ignore_iof_thresh, 0) and ignored_gt_boxes is not None:
        #     ignored_overlaps = unaligned_box_iof(proposals, ignored_gt_boxes)   # (n, m)
        #     ignored_max_overlaps = tf.reduce_max(ignored_overlaps, axis=1, keepdims=True)  # (n, )
        #
        #     ignored_mask = tf.greater_equal(ignored_max_overlaps, self.ignore_iof_thresh)
        #     overlaps *= tf.cast(ignored_mask, overlaps.dtype)

        target_boxes, target_labels, target_mask = self.assign_wrt_overlaps(
            overlaps, gt_boxes, gt_labels, feat_size, strides)
        target_mask = tf.expand_dims(target_mask, -1)
        
        return target_boxes, target_labels, target_mask

    def assign_wrt_overlaps(self, overlaps, gt_boxes, gt_labels, feat_size, strides):
        """Assign w.r.t. the overlaps of boxes with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_boxes and n proposals,
                shape (k, n).
            gt_boxes (Tensor): Ground-truth boxes, shape (k, 4).
            feat_size

        Returns:
            target_boxes (Tensor), target_labels (Tensor).
        """
        with tf.name_scope("assign_wrt_overlaps"):
            num_proposals = tf.shape(overlaps)[0]

            # 1. assign every proposal to -1
            target_labels = tf.cast(tf.fill([num_proposals], value=-1), gt_boxes.dtype)

            max_overlaps = tf.reduce_max(overlaps, axis=1)
            argmax_overlaps = tf.argmax(overlaps, axis=1)

            target_boxes = tf.gather(gt_boxes, indices=argmax_overlaps)

            # 2. assign negative
            negative = tf.logical_and(x=tf.greater_equal(max_overlaps, self.neg_iou_thresh[0]),
                                      y=tf.less(max_overlaps, self.neg_iou_thresh[1]))
            target_labels = tf.where(negative, tf.zeros_like(target_labels), target_labels)
            positive = tf.greater_equal(max_overlaps, self.pos_iou_thresh)
            target_labels = tf.where(positive, tf.gather(gt_labels, argmax_overlaps), target_labels)

            assigned_boxes_mask = tf.greater_equal(tf.reduce_max(overlaps, axis=0), self.pos_iou_thresh)
            assigned_boxes = tf.boolean_mask(gt_boxes, assigned_boxes_mask)
            mask = tf.fill(feat_size, value=0.)
            feat_size = tf.cast(feat_size, assigned_boxes.dtype)
            strides = tf.cast(strides, assigned_boxes.dtype)
            for i in tf.range(tf.shape(assigned_boxes)[0]):
                mask = self._assign_mask(assigned_boxes[i], mask, feat_size, strides)

            return target_boxes, target_labels, mask

    def _assign_mask(self, box, mask, feat_size, strides):
        y1 = tf.cast(tf.math.ceil(box[0] * feat_size[0]), tf.int64)
        x1 = tf.cast(tf.math.ceil(box[1] * feat_size[1]), tf.int64)
        y2 = tf.cast(tf.math.floor(box[2] * feat_size[0]), tf.int64)
        x2 = tf.cast(tf.math.floor(box[3] * feat_size[1]), tf.int64)
        y2 = tf.minimum(y2 + 1, tf.cast(feat_size[0], tf.int64))
        x2 = tf.minimum(x2 + 1, tf.cast(feat_size[1], tf.int64))
        x_inds, y_inds = tf.meshgrid(tf.range(y1, y2), tf.range(x1, x2))
        yx_inds = tf.reshape(tf.stack([y_inds, x_inds], axis=-1), [-1, 2])
        
        update_values = tf.tile(tf.constant([1.], dtype=tf.float32), [tf.shape(yx_inds)[0]])
    
        mask = tf.tensor_scatter_nd_update(mask, yx_inds, update_values)

        return mask

    def __call__(self, gt_boxes, gt_labels, proposals, feat_size, strides):
        with tf.name_scope("mask_max_iou_assiger"):
            return self.assign(gt_boxes, gt_labels, proposals, feat_size, strides)
