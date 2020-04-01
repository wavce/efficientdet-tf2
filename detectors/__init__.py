from .detector import Detector

from .fcos import FCOS

from .efficientdet import EfficientDet
from .efficient_fcos import EfficientFCOS


DETECTOR = {
    "fcos": FCOS,
    "efficientdet": EfficientDet,
    "efficient_fcos": EfficientFCOS,
}


def build_detector(name, **kwargs):
    return DETECTOR[name](**kwargs)


__all__ = ["FCOS",
           "Detector",
           "EfficientDet",
           "EfficientFCOS",
           "build_detector",]

