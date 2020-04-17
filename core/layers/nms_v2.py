import tensorflow as tf


def generate_topk_predictions(predicted_boxes, predicted_scores, max_predictions, num_classes):    
    batch_size = tf.shape(predicted_boxes)[0]

    predicted_scores_reshape = tf.reshape(predicted_scores, [batch_size, -1])
    _, topk_indices = tf.nn.top_k(predicted_scores_reshape, k=max_predictions)
    indices = topk_indices // num_classes
    classes = topk_indices % num_classes
  
    cls_indices = tf.stack([indices, classes], axis=2)
    class_outputs_after_topk = tf.gather_nd(predicted_scores, cls_indices, batch_dims=1)
    box_outputs_after_topk = tf.gather_nd(predicted_boxes, tf.expand_dims(indices, 2), batch_dims=1)

    return box_outputs_after_topk, class_outputs_after_topk, indices, classes


class BatchNonMaxSuppression(object):
    def __init__(self, iou_threshold, score_threshold, pre_nms_size, post_nms_size, num_classes, **kwargs):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.pre_nms_size = pre_nms_size
        self.post_nms_size = post_nms_size
        self.num_classes = num_classes
    
    def _nms_per_image(self, pred_boxes, pred_scores, classes):
        nmsed_boxes_list = []
        nmsed_classes_list = []
        nmsed_scores_list = []
        for c in range(self.num_classes):
            c_mask = tf.equal(classes, c)
            c_boxes = tf.boolean_mask(pred_boxes, c_mask)
            c_scores = tf.boolean_mask(pred_scores, c_mask)

            nmsed_inds = tf.image.non_max_suppression(
                c_boxes, c_scores, self.post_nms_size, 
                self.iou_threshold, self.score_threshold)
            nmsed_boxes = tf.gather(pred_boxes, nmsed_inds)
            nmsed_classes = tf.gather(classes, nmsed_inds)
            nmsed_scores = tf.gather(pred_scores, nmsed_inds)
            nmsed_boxes_list.append(nmsed_boxes)
            nmsed_scores_list.append(nmsed_scores)
            nmsed_classes_list.append(nmsed_classes)
        
        nmsed_boxes_outputs = tf.concat(nmsed_boxes_list, 0)
        nmsed_scores_outputs = tf.concat(nmsed_scores_list, 0)
        nmsed_classes_outputs = tf.concat(nmsed_classes_list, 0)

        return nmsed_boxes_outputs, nmsed_scores_outputs, nmsed_classes_outputs

    def __call__(self, predicted_boxes, predicted_scores):
        with tf.name_scope("non_max_suppression"):
            topk_boxes, topk_scores, _, classes = generate_topk_predictions(
                predicted_boxes, predicted_scores, self.pre_nms_size, self.num_classes)
            batch_size = tf.shape(predicted_boxes)[0]
            nmsed_boxes_ta = tf.TensorArray(predicted_boxes.dtype, batch_size, True)
            nmsed_scores_ta = tf.TensorArray(predicted_scores.dtype, batch_size, True)
            nmsed_classes_ta = tf.TensorArray(classes.dtype, batch_size, True)
            num_detections_ta = tf.TensorArray(tf.int32, batch_size, True)

            for i in tf.range(batch_size):
                output_boxes, output_scores, output_classes = self._nms_per_image(
                    topk_boxes[i], topk_scores[i], classes[i])
                
                num_pred = tf.size(output_scores)
                if num_pred > self.post_nms_size:
                    output_scores, inds = tf.nn.top_k(output_scores, k=self.pre_nms_size)
                    output_boxes = tf.gather(output_boxes, inds)
                    output_classes = tf.gather(output_classes, inds)
                else:
                    n = self.post_nms_size - num_pred
                    output_boxes = tf.concat([output_boxes, tf.zeros([n, 4], output_boxes.dtype)], 0)
                    output_scores = tf.concat([output_scores, tf.zeros([n], output_scores.dtype)], 0)
                    output_classes = tf.concat([output_classes, tf.zeros([n], output_classes.dtype)], 0)
                
                nmsed_boxes_ta = nmsed_boxes_ta.write(i, output_boxes)
                nmsed_scores_ta = nmsed_scores_ta.write(i, output_scores)
                nmsed_classes_ta = nmsed_classes_ta.write(i, output_classes)
                num_detections_ta = num_detections_ta.write(i, num_pred)

            return dict(nmsed_boxes=nmsed_boxes_ta.stack(),
                        nmsed_scores=nmsed_scores_ta.stack(),
                        nmsed_classes=nmsed_classes_ta.stack(),
                        valid_detections=num_detections_ta.stack())
                

class FastNonMaxSuppression(object):
    def __init__(self, cfg):
        self.cfg = cfg
    
    def unaligned_box_iou(self, boxes):
        """Calculate overlap between two set of unaligned boxes.
            'unaligned' mean len(boxes1) != len(boxes2).

            Args:
                boxes (tensor): shape (b, c, k, 4).
            Returns:
                ious (Tensor): shape (b, c, k, k)
        """
        boxes1 = boxes[:, :, :, None, :]   # (b, c, k, 4)
        boxes2 = boxes[:, :, None, :, :]   # (b, c, k, 4)
        lt = tf.maximum(boxes1[..., 0:2], boxes2[..., 0:2])  # (b, c, k, k, 2)
        rb = tf.minimum(boxes1[..., 2:4], boxes2[..., 2:4])  # (b, c, k, k, 2)

        wh = tf.maximum(0.0, rb - lt)  # (b, c, k, k, 2)
        overlap = tf.reduce_prod(wh, axis=4)  # (b, c, k, k)
        area1 = tf.reduce_prod(boxes1[..., 2:4] - boxes1[..., 0:2], axis=4)  # (b, c, k, k)
        area2 = tf.reduce_prod(boxes2[..., 2:4] - boxes2[..., 0:2], axis=4)

        ious = overlap / (area1 + area2 - overlap)

        return ious

    def __call__(self, predicted_boxes, predicted_scores):
        with tf.name_scope("fast_non_max_suppression"):
            max_predicted_scores = tf.reduce_max(predicted_scores, -1)

            k = self.cfg.postprocess.pre_nms_size
            # _, top_indices = tf.nn.top_k(max_predicted_scores, k=k)  # [b, n]
            batch_size = tf.shape(predicted_boxes)[0]
            # batch_inds = tf.tile(tf.expand_dims(tf.range(batch_size), -1), [1, k])
            # indices = tf.stack([batch_inds, top_indices], -1)
            # top_boxes = tf.gather_nd(predicted_boxes, indices)
            # top_scores = tf.gather_nd(predicted_scores, indices)

            thresholded_boxes, thresholded_scores = batch_threshold(
                predicted_boxes, predicted_scores, max_predicted_scores,
                self.cfg.postprocess.score_threshold, k)
            thresholded_scores = tf.transpose(thresholded_scores, [0, 2, 1])
            
            sorted_inds = tf.argsort(thresholded_scores, 2, "DESCENDING")
            batch_inds2 = tf.tile(tf.reshape(tf.range(batch_size), [batch_size, 1, 1]), [1, tf.shape(sorted_inds)[1], k])
            inds2 = tf.stack([batch_inds2, sorted_inds], -1)
            sorted_boxes = tf.gather_nd(thresholded_boxes, inds2)  # (b, c, k, 4)
            sorted_scores = tf.sort(thresholded_scores, 2, "DESCENDING")  # (b, c, k)
            ious = self.unaligned_box_iou(sorted_boxes)  # (b, c, k, k)
            ious = tf.linalg.band_part(ious, 0, -1) - tf.linalg.band_part(ious, 0, 0)  # (b, c, k, k)
            max_ious = tf.reduce_max(ious, 2)  # (b, c, k) 
            # Now just filter out the ones higher than the threshold
            keep = tf.less(max_ious, self.cfg.postprocess.iou_threshold) 
            # We should only keep detections over the confidence threshold
            keep = tf.logical_and(keep, tf.greater(sorted_scores, self.cfg.postprocess.score_threshold))  # (b, c, k)
            c = tf.shape(keep)[1]
            total_classes = tf.tile(tf.reshape(tf.range(c), [c, 1]), [1, tf.shape(keep)[2]])  # (c, k)

            nmsed_boxes_ta = tf.TensorArray(size=1, dynamic_size=True, dtype=predicted_boxes.dtype)
            nmsed_scores_ta = tf.TensorArray(size=1, dynamic_size=True, dtype=predicted_scores.dtype)
            nmsed_classes_ta = tf.TensorArray(size=1, dynamic_size=True, dtype=tf.int32)
            num_detections_ta = tf.TensorArray(size=1, dynamic_size=True, dtype=tf.int32) 
            for i in tf.range(batch_size):
                boxes = tf.boolean_mask(sorted_boxes[i], keep[i])
                scores = tf.boolean_mask(sorted_scores[i], keep[i])
                classes = tf.boolean_mask(total_classes, keep[i])
                num = tf.size(scores)
                max_total_size = self.cfg.postprocess.post_nms_size
                if tf.less(num, max_total_size):
                    boxes = tf.concat([boxes, tf.zeros([max_total_size - num, 4], boxes.dtype)], 0)
                    scores = tf.concat([scores, tf.zeros([max_total_size - num], scores.dtype)], 0)
                    classes = tf.concat([classes, -1 * tf.ones([max_total_size - num], classes.dtype)], 0)
                else:
                    boxes = boxes[:max_total_size]
                    scores = scores[:max_total_size]
                    classes = classes[:max_total_size]
                    num = tf.convert_to_tensor(max_total_size, num.dtype)
 
                nmsed_boxes_ta = nmsed_boxes_ta.write(i, boxes)
                nmsed_scores_ta = nmsed_scores_ta.write(i, scores)
                nmsed_classes_ta = nmsed_classes_ta.write(i, classes)
                num_detections_ta = num_detections_ta.write(i, num)

            nmsed_boxes = nmsed_boxes_ta.stack(name="nmsed_boxes")
            nmsed_scores = nmsed_scores_ta.stack(name="nmsed_scores")
            nmsed_classes = nmsed_classes_ta.stack(name="nmsed_classes")
            num_detections = num_detections_ta.stack(name="valid_detections")

            return dict(nmsed_boxes=nmsed_boxes,
                        nmsed_scores=nmsed_scores, 
                        nmsed_classes=nmsed_classes, 
                        valid_detections=num_detections)


class BatchSoftNonMaxSuppression(object):
    def __init__(self, iou_threshold, score_threshold, pre_nms_size, post_nms_size, num_classes, soft_nms_sigma=0.5, **kwargs):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.pre_nms_size = pre_nms_size
        self.post_nms_size = post_nms_size
        self.num_classes = num_classes
        self.soft_nms_sigma = soft_nms_sigma
    
    def _nms_per_image(self, pred_boxes, pred_scores, classes):
        nmsed_boxes_list = []
        nmsed_classes_list = []
        nmsed_scores_list = []
        for c in range(self.num_classes):
            c_mask = tf.equal(classes, c)
            c_boxes = tf.boolean_mask(pred_boxes, c_mask)
            c_scores = tf.boolean_mask(pred_scores, c_mask)

            nmsed_inds, nmsed_scores = tf.image.non_max_suppression_with_scores(
                c_boxes, c_scores, self.post_nms_size, self.iou_threshold, 
                self.score_threshold, self.soft_nms_sigma)
            nmsed_boxes = tf.gather(pred_boxes, nmsed_inds)
            nmsed_classes = tf.gather(classes, nmsed_inds)
            nmsed_boxes_list.append(nmsed_boxes)
            nmsed_scores_list.append(nmsed_scores)
            nmsed_classes_list.append(nmsed_classes)
        
        nmsed_boxes_outputs = tf.concat(nmsed_boxes_list, 0)
        nmsed_scores_outputs = tf.concat(nmsed_scores_list, 0)
        nmsed_classes_outputs = tf.concat(nmsed_classes_list, 0)

        return nmsed_boxes_outputs, nmsed_scores_outputs, nmsed_classes_outputs

    def __call__(self, predicted_boxes, predicted_scores):
        with tf.name_scope("non_max_suppression"):
            topk_boxes, topk_scores, _, classes = generate_topk_predictions(
                predicted_boxes, predicted_scores, self.pre_nms_size, self.num_classes)
            batch_size = tf.shape(predicted_boxes)[0]
            nmsed_boxes_ta = tf.TensorArray(predicted_boxes.dtype, batch_size, True)
            nmsed_scores_ta = tf.TensorArray(predicted_scores.dtype, batch_size, True)
            nmsed_classes_ta = tf.TensorArray(classes.dtype, batch_size, True)
            num_detections_ta = tf.TensorArray(tf.int32, batch_size, True)

            for i in tf.range(batch_size):
                output_boxes, output_scores, output_classes = self._nms_per_image(
                    topk_boxes[i], topk_scores[i], classes[i])
                
                num_pred = tf.size(output_scores)
                if num_pred > self.post_nms_size:
                    output_scores, inds = tf.nn.top_k(output_scores, k=self.pre_nms_size)
                    output_boxes = tf.gather(output_boxes, inds)
                    output_classes = tf.gather(output_classes, inds)
                else:
                    n = self.post_nms_size - num_pred
                    output_boxes = tf.concat([output_boxes, tf.zeros([n, 4], output_boxes.dtype)], 0)
                    output_scores = tf.concat([output_scores, tf.zeros([n], output_scores.dtype)], 0)
                    output_classes = tf.concat([output_classes, tf.zeros([n], output_classes.dtype)], 0)
                
                nmsed_boxes_ta = nmsed_boxes_ta.write(i, output_boxes)
                nmsed_scores_ta = nmsed_scores_ta.write(i, output_scores)
                nmsed_classes_ta = nmsed_classes_ta.write(i, output_classes)
                num_detections_ta = num_detections_ta.write(i, num_pred)

            return dict(nmsed_boxes=nmsed_boxes_ta.stack(),
                        nmsed_scores=nmsed_scores_ta.stack(),
                        nmsed_classes=nmsed_classes_ta.stack(),
                        valid_detections=num_detections_ta.stack())
                

class CombinedNonMaxSuppression(object):
    def __init__(self, iou_threshold, score_threshold, pre_nms_size, post_nms_size, num_classes, **kwargs):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.pre_nms_size = pre_nms_size
        self.post_nms_size = post_nms_size
        self.num_classes = num_classes

    def __call__(self, predicted_boxes, predicted_scores):
        with tf.name_scope("combined_non_max_suppression"):
            return tf.image.combined_non_max_suppression(
                boxes=tf.expand_dims(predicted_boxes, 2),
                scores=predicted_scores,
                max_output_size_per_class=self.post_nms_size,
                max_total_size=self.post_nms_size,
                iou_threshold=self.iou_threshold,
                score_threshold=self.score_threshold)
