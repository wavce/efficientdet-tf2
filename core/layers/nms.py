import tensorflow as tf


class BatchNonMaxSuppression(object):
    def __init__(self, cfg):
        self.cfg = cfg
    
    def __call__(self, predicted_boxes, predicted_scores):
        with tf.name_scope("non_max_suppression"):
            nmsed_boxes_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=predicted_boxes.dtype)
            nmsed_scores_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=predicted_scores.dtype)
            num_detections_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=tf.int32)
            
            for i in tf.range(tf.shape(predicted_boxes)[0]):
                # top_scores, top_indices = tf.nn.top_k(predicted_scores[i], 
                #                                       k=self.cfg.postprocess.pre_nms_size)
                # top_boxes = tf.gather(predicted_boxes[i], top_indices)

                post_nms_size = self.cfg.postprocess.post_nms_size
                selected_indices = tf.image.non_max_suppression(
                    boxes=predicted_boxes[i],
                    scores=predicted_scores[i],
                    max_output_size=post_nms_size,
                    iou_threshold=self.cfg.postprocess.iou_threshold,
                    score_threshold=self.cfg.postprocess.score_threshold)
                selected_boxes = tf.gather(predicted_boxes[i], selected_indices)
                selected_scores = tf.gather(predicted_scores[i], selected_indices)

                num = tf.size(selected_indices)
                if tf.less(num, self.cfg.postprocess.post_nms_size):
                    selected_boxes = tf.concat(
                        [selected_boxes, tf.zeros([post_nms_size - num, 4], selected_boxes.dtype)], 0)
                    selected_scores = tf.concat(
                        [selected_scores, tf.zeros([post_nms_size - num], selected_boxes.dtype)], 0)
                else:
                    selected_boxes = selected_boxes[:post_nms_size]
                    selected_scores = selected_scores[:post_nms_size]
                
                nmsed_boxes_ta = nmsed_boxes_ta.write(i, selected_boxes)
                nmsed_scores_ta = nmsed_scores_ta.write(i, selected_scores)
                num_detections_ta = num_detections_ta.write(i, num)
            
            return nmsed_boxes_ta.stack(), nmsed_scores_ta.stack(), num_detections_ta.stack()
                

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
            _, top_indices = tf.nn.top_k(max_predicted_scores, k=k)  # [b, n]
            batch_size = tf.shape(predicted_boxes)[0]
            batch_inds = tf.tile(tf.expand_dims(tf.range(batch_size), -1), [1, k])
            indices = tf.stack([batch_inds, top_indices], -1)
            top_boxes = tf.gather_nd(predicted_boxes, indices)
            top_scores = tf.gather_nd(predicted_scores, indices)
            top_scores = tf.transpose(top_scores, [0, 2, 1])
            
            sorted_inds = tf.argsort(top_scores, 2, "DESCENDING")
            batch_inds2 = tf.tile(tf.reshape(tf.range(batch_size), [batch_size, 1, 1]), [1, tf.shape(sorted_inds)[1], k])
            inds2 = tf.stack([batch_inds2, sorted_inds], -1)
            sorted_boxes = tf.gather_nd(top_boxes, inds2)  # (b, c, k, 4)
            sorted_scores = tf.sort(top_scores, 2, "DESCENDING")  # (b, c, k)
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
    def __init__(self, cfg):
        self.cfg = cfg
    
    def __call__(self, predicted_boxes, predicted_scores):
        with tf.name_scope("combined_soft_non_max_suppression"):
            nmsed_boxes_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=predicted_boxes.dtype)
            nmsed_scores_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=predicted_scores.dtype)
            num_detections_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=tf.int32)
            
            for i in tf.range(tf.shape(predicted_boxes)[0]):
                # top_scores, top_indices = tf.nn.top_k(predicted_scores[i], 
                #                                       k=self.cfg.postprocess.pre_nms_size)
                # top_boxes = tf.gather(predicted_boxes[i], top_indices)

                post_nms_size = self.cfg.postprocess.post_nms_size
                selected_indices, _ = tf.image.non_max_suppression_with_scores(
                    boxes=predicted_boxes[i],
                    scores=predicted_scores[i],
                    max_output_size=post_nms_size,
                    iou_threshold=self.cfg.postprocess.iou_threshold,
                    score_threshold=self.cfg.postprocess.score_threshold,
                    soft_nms_sigma=0.5)
                selected_boxes = tf.gather(predicted_boxes[i], selected_indices)
                selected_scores = tf.gather(predicted_scores[i], selected_indices)

                num = tf.size(selected_indices)
                if tf.less(num, self.cfg.postprocess.post_nms_size):
                    selected_boxes = tf.concat(
                        [selected_boxes, tf.zeros([post_nms_size - num, 4], selected_boxes.dtype)], 0)
                    selected_scores = tf.concat(
                        [selected_scores, tf.zeros([post_nms_size - num], selected_boxes.dtype)], 0)
                else:
                    selected_boxes = selected_boxes[:post_nms_size]
                    selected_scores = selected_scores[:post_nms_size]
                
                nmsed_boxes_ta = nmsed_boxes_ta.write(i, selected_boxes)
                nmsed_scores_ta = nmsed_scores_ta.write(i, selected_scores)
                num_detections_ta = num_detections_ta.write(i, num)
            
            return nmsed_boxes_ta.stack(), nmsed_scores_ta.stack(), num_detections_ta.stack()
                

class CombinedNonMaxSuppression(object):
    def __init__(self, cfg):
        
        self.cfg = cfg
    
    def __call__(self, predicted_boxes, predicted_scores):
        with tf.name_scope("combined_non_max_suppression"):
            # max_predicted_scores = tf.reduce_max(predicted_scores, -1)
            # _, top_indices = tf.nn.top_k(max_predicted_scores, k=self.cfg.postprocess.pre_nms_size)

            # batch_size = tf.shape(predicted_boxes)[0]
            # top_box_indices = tf.tile(tf.reshape(tf.range(batch_size), [batch_size, 1]), 
            #                         [1, self.cfg.postprocess.pre_nms_size])
            # top_box_indices = tf.stack([top_box_indices, top_indices], -1)
            # top_boxes = tf.gather_nd(predicted_boxes, top_box_indices)
            # top_scores = tf.gather_nd(predicted_scores, top_box_indices)

            return tf.image.combined_non_max_suppression(
                boxes=tf.expand_dims(predicted_boxes, 2),
                scores=predicted_scores,
                max_output_size_per_class=self.cfg.postprocess.post_nms_size,
                max_total_size=self.cfg.postprocess.post_nms_size,
                iou_threshold=self.cfg.postprocess.iou_threshold,
                score_threshold=self.cfg.postprocess.score_threshold)