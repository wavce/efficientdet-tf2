from .sampler import Sampler
from .ohem_sampler import OHEMSampler
from .pseudo_sampler import PseudoSampler
from .random_sampler import RandomSampler
from .combined_sampler import CombinedSampler


SAMPLERS = {
    "ohem_sampler": OHEMSampler,
    "pseudo_sampler": PseudoSampler,
    "random_sampler": RandomSampler,
    "combined_sampler": CombinedSampler
}


def build_sampler(**kwargs):
    sampler = kwargs["sampler"]
    if sampler == "combined_sampler":
        pos_sampler_kwargs = kwargs["pos_sampler"]
        neg_sampler_kwargs = kwargs["neg_sampler"]

        pos_sampler = SAMPLERS[pos_sampler_kwargs.pop["sampler"]](**pos_sampler_kwargs)
        neg_sampler = SAMPLERS[neg_sampler_kwargs.pop["sampler"]](**neg_sampler_kwargs)

        return CombinedSampler(pos_sampler, neg_sampler)
    
    return SAMPLERS[sampler](**kwargs)


__all__ = ["Sampler",
           "build_sampler"]
