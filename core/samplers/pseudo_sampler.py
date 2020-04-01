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
                A dict -> target_boxes, target_labels, positive_indices, label_inidces
        """
        pos_inds = tf.squeeze(tf.where(assigned_labels >= 1), 1)
        label_inds = tf.squeeze(tf.where(assigned_labels >= 0), 1)

        target_boxes = tf.gather(assigned_boxes, pos_inds)
        target_labels = tf.gather(assigned_labels, label_inds)

        return dict(target_boxes=target_boxes,
                    target_labels=target_labels,
                    positive_indices=pos_inds,
                    label_indices=label_inds,
                    num_pos=tf.size(pos_inds))
        