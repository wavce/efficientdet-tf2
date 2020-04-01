import tensorflow as tf
from core.samplers import Sampler


class OHEMSampler(Sampler):
    def __init__(self, num_proposals, pos_fraction, neg_pos_ub=-1, add_gt_as_proposals=True, **kwargs):
        super(OHEMSampler, self).__init__(num_proposals, pos_fraction, neg_pos_ub, add_gt_as_proposals)
    
    def _hard_mining(self, losses, indices, num):
        valid_losses = tf.gather(losses, indices)

        _, top_k_inds = tf.nn.top_k(valid_losses, k=num)

        return tf.stop_gradient(top_k_inds)

    def _sample_positive(self, assigned_labels, losses, num_expected_proposals, **kwargs):
        pos_inds = tf.where(assigned_labels >= 1)
        pos_inds = tf.squeeze(pos_inds, 1)
        if tf.size(pos_inds) <= num_expected_proposals:
            return pos_inds
        
        return self._hard_mining(losses, pos_inds, num_expected_proposals)
    
    def _sample_negative(self, assigned_labels, losses, num_expected_proposals, **kwargs):
        neg_inds = tf.where(assigned_labels == 0)
        neg_inds = tf.squeeze(neg_inds, 1)
        if tf.size(neg_inds) <= num_expected_proposals:
            return neg_inds
        
        return self._hard_mining(losses, neg_inds, num_expected_proposals)      
