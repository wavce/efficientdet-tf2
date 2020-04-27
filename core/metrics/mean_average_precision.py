import numpy as np 
import tensorflow  as tf 


class mAP(tf.keras.metrics.Metric):
    def __init__(self, num_classes, iou_threshold=0.5, area_ranges=None, use_voc07_metric=False, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)

        self.num_classes = num_classes
        self.area_ranges = area_ranges if area_ranges is not None else [(None, None)]
        self.iou_threshold = iou_threshold
        self.true_positives = {i: [] for i in range(num_classes)}
        self.false_positives = {i: [] for i in range(num_classes)}
        self.predicted_scores = {i: [] for i in range(num_classes)}
        self.num_gts = {i: [0] * len(area_ranges) for i in range(num_classes)}
        self.use_voc07_metric = use_voc07_metric
    
    def voc_ap(self, tp, fp, pred_scores, num_gts, use_voc07_metric=False):
        indices = np.argsort(-pred_scores)
        tp = tp[:, indices]
        fp = fp[:, indices]

        fp = np.cumsum(fp, 1)
        tp = np.cumsum(tp, 1)
        recall = tp / (num_gts[:, None] + 1)
        precision = tp / (fp + tp + np.finfo(np.float32).eps)

        no_scale = False
        if recall.ndim == 1:
            no_scale = True
            recall = recall[np.newaxis, :]
            precision = precision[np.newaxis, :]
        
        assert recall.shape == precision.shape and recall.ndim == 2
        num_scales = recall.shape[0]
        ap = np.zeros(num_scales, dtype=np.float32)
        
        if use_voc07_metric:
            # 11 point metric
            for i in range(num_scales):
                for t in np.arange(0.0, 1.1, 0.1):
                    prec = precision[recall >= t]
                    prec = prec.max() if prec.size > 0 else 0
                    ap[i] += prec
                ap[i] /= 11
        else:
            zeros = np.zeros((num_scales, 1), dtype=recall.dtype)
            ones = np.ones((num_scales, 1), dtype=recall.dtype)
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate((zeros, recall, ones), 1)
            mpre = np.concatenate((zeros, precision, zeros), 1)

            # compute the precision envelope
            for j in range(mpre.shape[1] - 1, 0, -1):
                mpre[:, j - 1] = np.maximum(mpre[:, j - 1], mpre[:, j])

            for i in range(num_scales):
                # to calculate area under PR curve, look for points
                # where X axis (recall) changes value
                inds = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
                # and sum (\Delta recall) * prec
                ap[i] = np.sum((mrec[i, inds + 1] - mrec[i, inds]) * mpre[i, inds + 1])

        return ap
    
    def unaligned_box_iou(self, boxes1, boxes2):
        """Calculate overlap between two set of unaligned boxes.
            'unaligned' mean len(boxes1) != len(boxes2).

            Args:
                boxes1 (tensor): shape (n, 4).
                boxes2 (tensor): shape (m, 4), m not equal n.

            Returns:
                ious (Tensor): shape (n, m)
        """
        boxes1 = boxes1[:, None, :]   # (n, 1, 4)
        boxes2 = boxes2[None, :, :]   # (1, m, 4)
        lt = np.maximum(boxes1[..., 0:2], boxes2[..., 0:2])  # (n, m, 2)
        rb = np.minimum(boxes1[..., 2:4], boxes2[..., 2:4])  # (n, m, 2)

        wh = np.maximum(0.0, rb - lt + 1)  # (n, m, 2)
        overlap = wh[..., 0] * wh[..., 1]  # (n, m)
        area1 = (boxes1[..., 2] - boxes1[..., 0] + 1) * (boxes1[..., 3] - boxes1[..., 1] + 1)  # (n, m)
        area2 = (boxes2[..., 2] - boxes2[..., 0] + 1) * (boxes2[..., 3] - boxes2[..., 1] + 1)  # (n, m)

        ious = overlap / (area1 + area2 - overlap)

        return ious
    
    def tpfp_default(self, 
                     pred_boxes, 
                     pred_scores,
                     gt_boxes, 
                     gt_boxes_ignore=None):
        """Check if predicted boxes are true positive or false positive.
        
        Args:
            pred_boxes (ndarray): Predicted boxes of this image, of shape (m, 4).
            gt_boxes (ndarray): GT boxes of this image, of shape (n, 4).
            gt_boxes_ignore (ndarray): Ignored gt boxes of this image, of shape (k, 4).
                Default to None.
            iou_threshold (float): IoU threshold to be considered as matched for 
                medium and large boxes (small ones have special rules).
            area_ranges (ndarray): Range of box areas to be evaluated, int the format
                [[min1, max1], [min2, max2], ...]. Defualt to None.
        
        Returns:
            tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of every 
                array is (num_scales, m).
        """
        # an indicator of ignored gts
        gt_ignore_inds = np.zeros(gt_boxes.shape[0], dtype=np.bool)
        if gt_boxes_ignore is not None:
            gt_ignore_inds = np.concatenate(
                (gt_ignore_inds, np.ones(gt_boxes_ignore.shape[0], dtype=np.bool)))

            # stack gt_boxes and gt_boxe_ignore for convience
            gt_boxes = np.vstack((gt_boxes, gt_boxes_ignore))
        
        num_preds = pred_boxes.shape[0]
        num_gts = gt_boxes.shape[0]
        area_ranges = self.area_ranges
        
        num_scales = len(area_ranges)
        # tp and fp are of shape (num_scales, num_preds), each row is tp or fp of certain scale.
        tp = np.zeros((num_scales, num_preds), dtype=np.float32)
        fp = np.zeros((num_scales, num_preds), dtype=np.float32)
        
        if num_gts == 0:
            if area_ranges == [(None, None)]:
                fp[...] = 1
            else:
                pred_areas = (pred_boxes[:, 2] - pred_boxes[:, 0] + 1.) * (pred_boxes[:, 3] - pred_boxes[:, 1] + 1.)
                for i, (min_area, max_area) in enumerate(area_ranges):
                    fp[i, (pred_areas >= min_area) & (pred_areas < max_area)] = 1
            
            return tp, fp

        ious = self.unaligned_box_iou(pred_boxes, gt_boxes)
        # for each pred, the max iou with all gts
        max_ious = ious.max(axis=1)
        max_inds = ious.argmax(axis=1)
        sort_inds = np.argsort(-pred_scores)
        for k, (min_area, max_area) in enumerate(area_ranges):
            gt_covered = np.zeros(num_gts, dtype=np.bool)
            # if no area range is specified, gt_area_ignore is all False
            if min_area is None:
                gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=np.bool)
            else:
                gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
                gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
            
            for i in sort_inds:
                if max_ious[i] >= self.iou_threshold:
                    matched_gt = max_inds[i]
                    if not (gt_ignore_inds[matched_gt] or gt_area_ignore[matched_gt]):
                        if not gt_covered[matched_gt]:
                            gt_covered[matched_gt] = True
                            tp[k, i] = 1
                        else:
                            fp[k, i] = 1
                    # otherwise ignore this predicted box, tp = 0, fp = 0
                elif min_area is None:
                    fp[k, i] = 1
                else:
                    bbox = pred_boxes[i, :]
                    area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
                    if area >= min_area and area < max_area:
                        fp[k, i] = 1
        
        return tp, fp

    def update_state(self, 
                     gt_boxes, 
                     gt_classes, 
                     pred_boxes, 
                     pred_scores, 
                     pred_classes, 
                     area_ranges=None,
                     gt_boxes_ignore=None, 
                     sample_weights=None):
        batch_size = tf.shape(gt_boxes)[0]
        for i in range(self.num_classes):
            for b in tf.range(batch_size):
                valid_gt_mask = tf.logical_not(tf.reduce_all(gt_boxes[b] <= 0, 1))
                valid_pred_mask = tf.logical_not(tf.reduce_all(pred_boxes[b] <= 0, 1))

                valid_gt_boxes = tf.boolean_mask(gt_boxes[b], valid_gt_mask)
                valid_gt_classes = tf.boolean_mask(gt_classes[b], valid_gt_mask)
                valid_pred_boxes = tf.boolean_mask(pred_boxes[b], valid_pred_mask)
                valid_pred_classes = tf.boolean_mask(pred_classes[b], valid_pred_mask)
                valid_pred_scores = tf.boolean_mask(pred_scores[b], valid_pred_mask)

                cls_gt_mask = valid_gt_classes == i + 1
                cls_pred_mask = valid_pred_classes == i + 1
                cls_gt_boxes = tf.boolean_mask(valid_gt_boxes, cls_gt_mask)
                # cls_gt_classes = tf.boolean_mask(valid_gt_classes, cls_gt_mask)
                cls_pred_boxes = tf.boolean_mask(valid_pred_boxes, cls_pred_mask)
                cls_pred_scores = tf.boolean_mask(valid_pred_scores, cls_pred_mask)
                # cls_pred_classes = tf.boolean_mask(valid_pred_classes, cls_pred_mask)

                if tf.reduce_any(cls_pred_mask):
                    tp, fp = tf.numpy_function(
                        func=self.tpfp_default,
                        inp=(cls_pred_boxes, cls_pred_scores, cls_gt_boxes),
                        Tout=(tf.float32, tf.float32))
                    # tp, fp = self.tpfp_default(cls_pred_boxes.numpy(), cls_pred_scores.numpy(), cls_gt_boxes.numpy())

                    if self.area_ranges == [(None, None)]:
                        self.num_gts[i][0] += tf.reduce_sum(tf.cast(cls_gt_mask, tf.float32))
                    else:
                        gt_areas = ((cls_gt_boxes[:, 2] - cls_gt_boxes[:, 0] + 1) * 
                                    (cls_gt_boxes[:, 3] - cls_gt_boxes[:, 1] + 1))
                        for k, (min_area, max_area) in enumerate(self.area_ranges):
                            self.num_gts[i][k] += tf.reduce_sum(
                                tf.cast(tf.logical_and(gt_areas >= min_area, gt_areas < max_area), tf.float32))

                    self.true_positives[i].append(tp)
                    self.false_positives[i].append(fp)
                    self.predicted_scores[i].append(cls_pred_scores)

    def result(self):
        aps = []
        for i in range(self.num_classes):
            pred_scores = tf.concat(self.predicted_scores[i], 0)
            tp = tf.concat(self.true_positives[i], 1)
            fp = tf.concat(self.false_positives[i], 1)
            num_gts = tf.convert_to_tensor(self.num_gts[i], tp.dtype)
            ap = tf.numpy_function(
                func=self.voc_ap,
                inp=(tp, fp, pred_scores, num_gts, self.use_voc07_metric),
                Tout=tf.float32)
            
            aps.append(ap)
        
        tf.print(aps)
        return tf.reduce_mean(aps, 0)
    
    def reset_states(self):
        del self.predicted_scores
        del self.true_positives
        del self.false_positives

        self.true_positives = {i: [] for i in range(self.num_classes)}
        self.false_positives = {i: [] for i in range(self.num_classes)}
        self.predicted_scores = {i: [] for i in range(self.num_classes)}
        self.num_gts = {i: [0] * len(area_ranges) for i in range(self.num_classes)}
        
        super(mAP, self).reset_states()