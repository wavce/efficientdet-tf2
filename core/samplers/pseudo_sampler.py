import tensorflow as tf
from core.samplers import Sampler


class PseudoSampler(Sampler):
    def __init__(self, **kwargs):
        pass
    
    def _sample_positive(self, assigned_labels, num_expected_proposals, **kwargs):
        raise NotImplementedError
    
    def _sample_negative(self, assigned_labels, num_expected_proposals, **kwargs):
        raise NotImplementedError

    def sample(self, assigned_boxes, assigned_labels, **kwargs):
        """Sample positive and negative boxes.

            Args:
                assigned_boxes (Tensor): The assigned boxes in assigner.
                assigned_labels (Tensor): The assigned labels in assigner.
            
            Returns:
                A dict -> target_boxes, target_labels, box_weights, label_weights
        """
        pos_mask = assigned_labels >= 1
        
        box_weights = tf.expand_dims(tf.cast(pos_mask, assigned_boxes.dtype), -1)
        valid_mask = assigned_labels >= 0
        target_labels = tf.where(valid_mask, assigned_labels, tf.zeros_like(assigned_labels))
        label_weights = tf.cast(valid_mask, assigned_labels.dtype)

        return dict(target_boxes=assigned_boxes,
                    target_labels=target_labels,
                    box_weights=box_weights,
                    label_weights=label_weights,
                    num_pos=tf.reduce_sum(box_weights))
        