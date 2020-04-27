from .preprocess import should_apply_op

from .preprocess import Rotate
from .preprocess import SSDCrop
from .preprocess import SFDetCrop
from .preprocess import RetinaCrop
from .preprocess import FlipLeftToRight
# from .preprocess import AutoAugmentation
from .preprocess import DataAnchorSampling
from .preprocess import RandomDistortColor


AUGS = {
    "rotate": Rotate,
    "flip_left_to_right": FlipLeftToRight,
    "random_distort_color": RandomDistortColor
}


CROPS = {
    "ssd_crop": SSDCrop,
    "sfdet_crop": SFDetCrop,
    "retina_crop": RetinaCrop,
    "data_anchor_sampling": DataAnchorSampling,
}


class Compose(object):
    def __init__(self, input_size, aug_cfgs):
        self.aug_cfgs = aug_cfgs

        self.input_size = input_size
        self.crop_cfgs = [cfg for cfg in aug_cfgs if list(cfg.keys())[0] in CROPS]
        
        self.other_cfgs = [cfg for cfg in aug_cfgs if list(cfg.keys())[0] not in CROPS]
    
    def _combined_crops(self, image, boxes, labels):
        assert len(self.crop_cfgs) == 2, "Only support to combining two crop."
        crop_ops = [list(cfg.values())[0] for cfg in self.crop_cfgs]
        prob1 = crop_ops[0].pop("probability")
        prob2 = crop_ops[1].pop("probability")
       
        assert prob1 + prob2 == 1, "If using crop op, the sum of probabilities of the two ops must be 1."
        if should_apply_op(prob1):
            crop_type = list(self.crop_cfgs[0].keys())[0]
            aug_kwargs = self.crop_cfgs[0][crop_type]
            image, boxes, labels = CROPS[crop_type](
                input_size=self.input_size, **aug_kwargs)(image, boxes, labels)
        else:
            crop_type = list(self.crop_cfgs[1].keys())[0]
            aug_kwargs = self.crop_cfgs[1][crop_type]
            image, boxes, labels = CROPS[crop_type](
                input_size=self.input_size, **aug_kwargs)(image, boxes, labels)

        return image, boxes, labels

    def __call__(self, image, boxes, labels):
        for cfg in self.other_cfgs:
            aug_type = list(cfg.keys())[0]
            aug_kwargs = cfg[aug_type]
            image, boxes, labels = AUGS[aug_type](**aug_kwargs)(image, boxes, labels)

        if len(self.crop_cfgs) > 1:
            image, boxes, labels = self._combined_crops(image, boxes, labels)
        else:
            crop_type = list(self.crop_cfgs[0].keys())[0]
            aug_kwargs = self.crop_cfgs[0][crop_type]
            aug_kwargs.pop("probability")
            image, boxes, labels = CROPS[crop_type](input_size=self.input_size, **aug_kwargs)(image, boxes, labels)

        return image, boxes, labels
