from .detector import Detector

from .fcos import FCOS

from .efficientdet import EfficientDet
from .efficient_fcos import EfficientFCOS


DETECTOR = {
    "fcos": FCOS,
    "efficientdet": EfficientDet,
    "efficient_fcos": EfficientFCOS,
}


def build_detector(detector, **kwargs):
    if detector.startswith("efficientdet"):
        name = "efficientdet"
        return DETECTOR[name](**kwargs)
    
    if detector.startswith("efficientdet") and detector.endswith("fcos"):
        name = "effcientdet_fcos"
        return DETECTOR[name](**kwargs)
    
    return DETECTOR[detector](**kwargs)


__all__ = ["FCOS",
           "Detector",
           "EfficientDet",
           "EfficientFCOS",
           "build_detector",]

