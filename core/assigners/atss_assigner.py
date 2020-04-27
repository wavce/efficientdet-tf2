import tensorflow as tf 
from core.bbox import unaligned_box_iou


class ATSSAssigner(object):
    """Assign a corresponding bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index(1-based) of assigned gt

    Args:
        topk (float): number of bbox selected in each level
    """
    def __init__(self, topk, num_proposals_per_level):
        self.topk = topk
        self.num_proposals_per_level = num_proposals_per_level
    
    def assign(self, gt_boxes, gt_labels, proposals, gt_boxes_ignore=None):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute iou between all proposal and gt_box
        2. compute center distance between all proposal and gt_box
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center.
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as postive
        6. limit the positive sample's center in gt_box

        Args:
            gt_boxes: Groundtruth boxes, shape (k, 4).
            gt_labels: Label of gt_boxes, shape (k, ).
            proposals: Bounding boxes to be assigned, shape(n, 4).
            gt_boxes_ignore: Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
        Returns:
            target_boxes, target_labels
        """
        INF = 100000000

        # 1. compute iou between all proposal and gt_box
        overlaps = unaligned_box_iou(gt_boxes, proposals)   # [k, n]
        num_gts = tf.shape(overlaps)[0]    # [k, ]
        num_proposals = tf.shape(overlaps)[1]   #[n, ]

        if num_gts == 0:
            return (tf.zeros([num_proposals, 4], tf.float32), 
                    tf.zeros([num_proposals], gt_labels.dtype))
        
        # 2. compute center distance between all proposal and gt_box
        gt_centers = (gt_boxes[:, 0:2] + gt_boxes[:, 2:4]) * 0.5
        proposal_centers = (proposals[:, 0:2] + proposals[:, 2:4]) * 0.5
        distances = tf.math.sqrt(
            tf.reduce_sum(
                tf.math.squared_difference(
                    gt_centers[:, None, :], proposal_centers[None, :, :]), -1))  # (k, n)
        # 3. on each pyramid level, for each gt, select k bbox whose center
        #    are closest to the gt center.
        topk_inds_list = []
        start_ind = 0
        for level, num in enumerate(self.num_proposals_per_level):
            end_ind = start_ind + num
            _, topk_inds = tf.nn.top_k(-distances[:, start_ind:end_ind], k=self.topk)   # (k, topk)
            topk_inds_list.append(topk_inds + start_ind)
            start_ind = end_ind
        
        topk_inds = tf.concat(topk_inds_list, 1)
        num_topk = self.topk * len(self.num_proposals_per_level)
        inds = tf.stack([tf.reshape(tf.repeat(tf.range(num_gts), num_topk), [num_gts, num_topk]), topk_inds], -1)
        # 4. get corresponding iou for the these candidates, and compute the
        #    mean and std, set mean + std as the iou threshold
        candidate_overlaps = tf.gather_nd(overlaps, inds)   # [k, topk]
        mean_per_gt, var_per_gt = tf.nn.moments(candidate_overlaps, 1, keepdims=True)
        std_per_gt = tf.math.sqrt(var_per_gt)
        overlaps_thresh_per_gt = mean_per_gt + std_per_gt
        
        # 5. select these candidates whose iou are greater than or equal to
        #    the threshold as postive
        is_pos = candidate_overlaps >= overlaps_thresh_per_gt  # (k, topk)
        # 6. limit the positive sample's center in gt_boxes
        
        # calculate the left, top, right, bottom distance between 
        # positive box center and gt_box side
        left = tf.tile(tf.expand_dims(proposal_centers[:, 0], 0), [num_gts, 1]) - gt_boxes[:, 0:1]  # (k, n)
        top = tf.tile(tf.expand_dims(proposal_centers[:, 1], 0), [num_gts, 1]) - gt_boxes[:, 1:2]
        right = gt_boxes[:, 2:3] - tf.tile(tf.expand_dims(proposal_centers[:, 0], 0), [num_gts, 1])
        bottom = gt_boxes[:, 3:4] - tf.tile(tf.expand_dims(proposal_centers[:, 1], 0), [num_gts, 1])
        is_in_gt = tf.reduce_min(tf.stack([left, top, right, bottom], -1), -1) > 0.01   # (k, n)
        is_in_gt = tf.gather_nd(is_in_gt, inds)    
        is_pos = tf.logical_and(is_pos, is_in_gt)  

        topk_inds += (tf.reshape(tf.repeat(tf.range(num_gts), num_topk), [num_gts, num_topk]) * num_proposals)
        candidate_inds = tf.boolean_mask(topk_inds, is_pos)

        # if an anchor box is assigned to multiple gts
        # the one with highest IoU will be seleted.
        overlaps_inf = tf.cast(tf.fill([num_gts * num_proposals], -INF), tf.float32)
        overlaps_inf = tf.tensor_scatter_nd_update(
            overlaps_inf, candidate_inds[:, None], tf.gather(tf.reshape(overlaps, [num_gts * num_proposals]), candidate_inds))
        
        overlaps_inf = tf.reshape(overlaps_inf, (num_gts, num_proposals))
        max_overlaps = tf.reduce_max(overlaps_inf, 0)
        argmax_overlaps = tf.argmax(overlaps_inf, 0)

        target_boxes = tf.gather(gt_boxes, argmax_overlaps)
        target_labels = tf.gather(gt_labels, argmax_overlaps)
        target_labels = tf.where(max_overlaps > -INF, target_labels, tf.zeros_like(target_labels))
 
        return target_boxes, target_labels
    
    def __call__(self, gt_boxes, gt_labels, proposals, gt_boxes_ignore=None):
        with tf.name_scope("atss_assigner"):
            return self.assign(gt_boxes, gt_labels, proposals, gt_boxes_ignore)
        
