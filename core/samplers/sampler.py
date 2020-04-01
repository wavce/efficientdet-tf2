import tensorflow as tf
from abc import ABCMeta
from abc import abstractmethod


class Sampler(metaclass=ABCMeta):
    def __init__(self, num_proposals, pos_fraction, neg_pos_ub=-1, add_gt_as_proposals=True, **kwargs):
        self.num_proposals = num_proposals
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals

        self.positive_sampler = self
        self.negative_sampler = self
    
    @abstractmethod
    def _sample_positive(self, assigned_labels, num_expected_proposals, **kwargs):
        pass
    
    @abstractmethod
    def _sample_negative(self, assigned_labels, num_expected_proposals, **kwargs):
        pass

    def sample(self, assigned_boxes, assigned_labels, gt_boxes, gt_labels=None, **kwargs):
        """Sample positive and negative boxes.

            Args:
                assigned_boxes (Tensor): The assigned boxes in assigner.
                assigned_labels (Tensor): The assigned labels in assigner.
                gt_boxes (Tensor): ground truth boxes.
                gt_labels (Tensor): ground truth labels.
            
            Returns:
                A dict -> target_boxes, target_labels, positive_indices, label_inidces
        """
        if self.add_gt_as_proposals:
            assigned_boxes = tf.concat([gt_boxes, assigned_boxes], 0)
            assigned_labels = tf.concat([gt_labels, assigned_labels], 0)
        
        num_expected_pos = int(self.num_proposals * self.pos_fraction)
        pos_inds = self.positive_sampler._sample_positive(assigned_labels, num_expected_pos, **kwargs)
        num_sampled_pos = tf.size(pos_inds)
        num_expected_neg = self.num_proposals - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = tf.maximum(1, num_expected_pos)
            neg_upper_bound = num_expected_neg * _pos

            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.negative_sampler._sample_negative(assigned_labels, num_expected_neg, **kwargs)

        target_boxes = tf.gather(assigned_boxes, pos_inds)

        label_inds = tf.concat([pos_inds, neg_inds], 0)
        target_labels = tf.gather(assigned_labels, label_inds)

        return dict(target_boxes=target_boxes,
                    target_labels=target_labels,
                    positive_indices=pos_inds,
                    label_indices=label_inds,
                    num_pos=tf.size(pos_inds))

