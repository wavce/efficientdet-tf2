import tensorflow as tf 


class ProposalsLayer(tf.keras.layers.Layer):
    def __init__(self, 
                 pre_nms_size=12000, 
                 post_nms_size=2000, 
                 max_total_size=2000, 
                 iou_threshold=0.7, 
                 use_sigmoid=False, 
                 name="proposals_layer",
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.pre_nms_size = pre_nms_size
        self.post_nms_size = post_nms_size
        self.iou_threshold = iou_threshold
        self.use_sigmoid = use_sigmoid

    def call(self, boxes, labels):
        if self.use_sigmoid:
            scores = tf.nn.sigmoid(labels)
        else:
            scores = tf.nn.softmax(labels)
        max_predicted_scores = tf.reduce_max(scores[..., 1:], -1)
        top_scores, top_indices = tf.nn.top_k(max_predicted_scores, k=self.pre_nms_size)

        batch_size = tf.shape(boxes)[0]
        top_box_indices = tf.tile(tf.reshape(tf.range(batch_size), [batch_size, 1]), 
                                    [1, self.cfg.postprocess.pre_nms_size])
        top_box_indices = tf.stack([top_box_indices, top_indices], -1)
        top_boxes = tf.gather_nd(boxes, top_box_indices)

        nmsed_boxes, _, _, _ = tf.image.combined_non_max_suppression(
                boxes=tf.expand_dims(top_boxes, 2),
                scores=top_scores,
                max_output_size_per_class=self.post_nms_size,
                max_total_size=self.post_nms_size,
                iou_threshold=self.iou_threshold)
        
        return nmsed_boxes    
