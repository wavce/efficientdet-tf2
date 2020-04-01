import tensorflow as tf
from core.samplers import Sampler


class CombinedSampler(Sampler):
    def __init__(self, pos_sampler, neg_sampler, **kwargs):
        super(CombinedSampler, self).__init__(**kwargs)

        self.positive_sampler = pos_sampler
        self.negative_sampler = neg_sampler
    
    def _sample_positive(self, assigned_labels, num_expected_proposals, **kwargs):
        raise NotImplementedError
    
    def _sample_negative(self, assigned_labels, num_expected_proposals, **kwargs):
        raise NotImplementedError